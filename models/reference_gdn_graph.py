"""CUDA-graph replay of the original GDN triangular loop, without changing its math.

The reference function is transformed at one verified AST loop only. All other
operations, including the original BF16 Q/K normalization, stay unchanged.
"""
import ast
from contextvars import ContextVar
import inspect
import logging
import textwrap

import torch

_ACTIVE=ContextVar('giga_reference_gdn_graph',default=None)


def triangular_reference(attn):
    for i in range(1,attn.shape[-1]):
        row=attn[...,i,:i].clone()
        sub=attn[...,:i,:i].clone()
        attn[...,i,:i]=row+(row.unsqueeze(-1)*sub).sum(-2)
    return attn


class ReferenceGDNGraphCache:
    def __init__(self,max_graphs=8,max_chunks=8):
        self.max_graphs=max(0,int(max_graphs));self.max_chunks=max(1,int(max_chunks))
        self.graphs={};self.hits=0;self.misses=0;self.fallbacks=0

    def stats(self):
        return {'graphs':len(self.graphs),'hits':self.hits,'misses':self.misses,'fallbacks':self.fallbacks}

    def __call__(self,attn):
        if not attn.is_cuda or torch.is_grad_enabled() or attn.shape[-2:]!=(64,64) or attn.shape[-3]>self.max_chunks:
            self.fallbacks+=1;return triangular_reference(attn)
        key=(attn.device.index,tuple(attn.shape),tuple(attn.stride()),attn.dtype)
        # Normal buffers are required even when the frozen encoder runs in inference mode.
        with torch.inference_mode(False),torch.no_grad():
            if key not in self.graphs:
                if len(self.graphs)>=self.max_graphs:
                    self.fallbacks+=1;return triangular_reference(attn.clone())
                self.misses+=1
                static=torch.empty_like(attn)
                stream=torch.cuda.Stream(device=attn.device)
                current=torch.cuda.current_stream(attn.device)
                stream.wait_stream(current)
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        static.copy_(attn);triangular_reference(static)
                stream.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream,capture_error_mode='thread_local'):
                    triangular_reference(static)
                current.wait_stream(stream)
                self.graphs[key]=(graph,static,stream)
            else:self.hits+=1
            graph,static,_=self.graphs[key]
            static.copy_(attn);graph.replay()
            # Each GDN layer must own its result while later layers replay the same graph.
            return static.clone()


def install_reference_graph_dispatch():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    if hasattr(modeling,'_giga_reference_graph_dispatch'):return
    previous=modeling.torch_chunk_gated_delta_rule
    reference=inspect.unwrap(previous)
    tree=ast.parse(textwrap.dedent(inspect.getsource(reference)))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
    matches=[]
    for node in function.body:
        if (isinstance(node,ast.For) and isinstance(node.iter,ast.Call)
            and isinstance(node.iter.func,ast.Name) and node.iter.func.id=='range'
            and len(node.iter.args)==2 and isinstance(node.iter.args[0],ast.Constant)
            and node.iter.args[0].value==1 and isinstance(node.iter.args[1],ast.Name)
            and node.iter.args[1].id=='chunk_size'):
            matches.append(node)
    if len(matches)!=1:raise RuntimeError('Installed GDN reference structure changed; refusing graph optimization')
    replacement=ast.parse('attn = _giga_triangular_graph(attn)').body[0]
    function.body=[replacement if node is matches[0] else node for node in function.body]
    function.decorator_list=[];function.name='_giga_reference_chunk_with_graph'
    namespace=dict(reference.__globals__)
    namespace['_giga_triangular_graph']=lambda attn:_ACTIVE.get()(attn)
    ast.fix_missing_locations(tree)
    exec(compile(tree,inspect.getfile(reference),'exec'),namespace)
    optimized=namespace[function.name]
    def dispatch(*args,**kwargs):
        if _ACTIVE.get() is None:return previous(*args,**kwargs)
        return optimized(*args,**kwargs)
    modeling.torch_chunk_gated_delta_rule=dispatch
    modeling._giga_reference_graph_dispatch=True


def activate(cache):
    return _ACTIVE.set(cache)


def deactivate(token):
    _ACTIVE.reset(token)
