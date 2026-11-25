"""mHC residual topology on the current Giga DiT, retaining all dense state keys.

The transformer body reuses current attention/MLP helpers (SiLU, dual/basis
AdaLN, FA3, MoE), and retains text refiner and final head modulation unchanged.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from models.giga_transformer import (GigaExpTransformer,GigaBlock,MAX_SEQLEN_IMAGE,
                                    MAX_SEQLEN_VIDEO,_FA3)
from models.moe import MoEFFN
from models.mhc_residual_transfer import MHCResidual,MHC_STAT_KEYS,read_out,stack_stream

MHC_VERSION='giga_mhc_v2_mean_transfer'


def is_mhc_parameter_name(name):
    return '.mhc_attn.' in name or '.mhc_ffn.' in name or name=='readout'


class GigaBlockMHC(GigaBlock):
    @classmethod
    def from_dense(cls,block,n_streams,sinkhorn_iters,mapping_fp32,init_mode,routing_backend='eager'):
        # Pure Python nn.Module subclass upgrade: no duplicate 5B model allocation,
        # no wrapping prefix or renaming of existing pretrained parameter keys.
        block.__class__=cls
        dim=block.qkv.in_features
        block.mhc_attn=MHCResidual(dim,n_streams,sinkhorn_iters,
                                  mapping_fp32=mapping_fp32,init_mode=init_mode,routing_backend=routing_backend)
        block.mhc_ffn=MHCResidual(dim,n_streams,sinkhorn_iters,
                                 mapping_fp32=mapping_fp32,init_mode=init_mode,routing_backend=routing_backend)
        return block

    def _attention_core(self,x,cond,cos,sin,attention_mask,pack,n_img):
        if self.mod_adaln:
            si,sci,gi,*_=self.mod_img(cond).chunk(6,-1)
            st,sct,gt,*_=self.mod_txt(cond).chunk(6,-1)
            image,text=x[:,:n_img],x[:,n_img:]
            y=torch.cat((self.norm1(image)*(1+sci[:,None])+si[:,None],
                         self.norm1(text)*(1+sct[:,None])+st[:,None]),1)
            out=self.out(self._attend(y,cos,sin,attention_mask,pack))
            return torch.cat((out[:,:n_img]*gi[:,None],out[:,n_img:]*gt[:,None]),1)
        shift,scale,gate,*_=self.mod(cond).chunk(6,-1)
        y=self.norm1(x)*(1+scale[:,None])+shift[:,None]
        return self.out(self._attend(y,cos,sin,attention_mask,pack))*gate[:,None]

    def _feedforward_core(self,x,cond,attention_mask,n_img,denoising_step):
        if self.mod_adaln:
            _,_,_,si,sci,gi=self.mod_img(cond).chunk(6,-1)
            _,_,_,st,sct,gt=self.mod_txt(cond).chunk(6,-1)
            image,text=x[:,:n_img],x[:,n_img:]
            y=torch.cat((self.norm2(image)*(1+sci[:,None])+si[:,None],
                         self.norm2(text)*(1+sct[:,None])+st[:,None]),1)
            gate=(torch.cat((gi[:,None].expand(x.shape[0],n_img,-1),
                             gt[:,None].expand(x.shape[0],x.shape[1]-n_img,-1)),1)
                  if isinstance(self.ff,MoEFFN) else None)
            out=self._ff(y,attention_mask,n_img,denoising_step,gate,residual=x)
            return torch.cat((out[:,:n_img]*gi[:,None],out[:,n_img:]*gt[:,None]),1)
        _,_,_,shift,scale,gate=self.mod(cond).chunk(6,-1)
        y=self.norm2(x)*(1+scale[:,None])+shift[:,None]
        return self._ff(y,attention_mask,n_img,denoising_step,gate[:,None],residual=x)*gate[:,None]

    def forward(self,h,cond,cos,sin,attention_mask=None,pack=None,n_img=None,denoising_step=None):
        if self.mod_adaln and (n_img is None or not 0<n_img<h.shape[1]):
            raise ValueError('dual/basis AdaLN needs a valid image/text split')
        h=self.mhc_attn(h,lambda x:self._attention_core(x,cond,cos,sin,attention_mask,pack,n_img))
        return self.mhc_ffn(h,lambda x:self._feedforward_core(x,cond,attention_mask,n_img,denoising_step))


class GigaExpTransformerMHC(GigaExpTransformer):
    is_mhc_model=True
    mhc_stat_keys=MHC_STAT_KEYS

    def __init__(self,*args,n_streams=4,sinkhorn_iters=20,mhc_mapping_fp32=True,
                 init_mode='transfer',routing_backend='eager',**kwargs):
        super().__init__(*args,**kwargs)
        self.n_streams=int(n_streams)
        self.mhc_init_mode=init_mode
        self.mhc_sinkhorn_iters=int(sinkhorn_iters)
        self.mhc_mapping_fp32=bool(mhc_mapping_fp32)
        for i,block in enumerate(self.blocks):
            self.blocks[i]=GigaBlockMHC.from_dense(block,self.n_streams,self.mhc_sinkhorn_iters,
                                                  self.mhc_mapping_fp32,init_mode,routing_backend)
        self.readout=nn.Parameter(torch.full((self.n_streams,),1/self.n_streams))
        self.mhc_routing_backend=routing_backend
        self._mhc_collect_stats=False
        self.last_migration_report=None

    def set_mhc_routing_backend(self, backend):
        for module in self.modules():
            if isinstance(module,MHCResidual):module.set_routing_backend(backend)
        self.mhc_routing_backend=backend

    def set_mhc_collect_stats(self,enabled):
        self._mhc_collect_stats=bool(enabled)
        for block in self.blocks:
            for name in ('mhc_attn','mhc_ffn'):
                module=getattr(block,name)
                module.collect_stats=bool(enabled)
                module.last_stats=None

    def cast_mhc_fp32(self):
        # Optional storage conversion only; callers using mixed storage must group
        # optimizer parameters by dtype. Not required for FP32 mapping arithmetic.
        for block in self.blocks:
            block.mhc_attn.float();block.mhc_ffn.float()
        self.readout.data=self.readout.data.float()

    def mhc_stats(self):
        if not self._mhc_collect_stats:return {}
        stats={}
        for i,block in enumerate(self.blocks):
            for name in ('mhc_attn','mhc_ffn'):
                for key,value in (getattr(block,name).last_stats or {}).items():
                    stats[f'blocks.{i}.{name}/{key}']=value
        r=self.readout.detach().float()
        for key,value in {'mean':r.mean(),'min':r.amin(),'max':r.amax(),'sum':r.sum(),'rms':r.square().mean().sqrt()}.items():
            stats[f'readout/{key}']=value
        return stats

    def semantic_metadata(self):
        metadata=super().semantic_metadata()
        metadata['mhc']={'version':MHC_VERSION,'n_streams':self.n_streams,
                         'sinkhorn_iters':self.mhc_sinkhorn_iters,'mapping_fp32':self.mhc_mapping_fp32,
                         'init_mode':self.mhc_init_mode,'input':'replicated','readout':'learned_mean_init',
                         'transfer_post':'asymmetric_mean_one'}
        return metadata

    def forward(self,x,timestep,text,text_mask=None,feature_layers=None):
        if x.ndim!=5:raise ValueError('x must be [B,C,T,H,W]')
        b,_,t,h,w=x.shape
        pt,ph,pw=self.patch_size
        if t%pt or h%ph or w%pw:raise ValueError('latent shape is not divisible by patch_size')
        t,h,w=t//pt,h//ph,w//pw
        image=x.reshape(b,self.in_channels,t,pt,h,ph,w,pw)
        image=self.patch_in(image.permute(0,2,4,6,3,5,7,1).reshape(b,t*h*w,-1))
        text=self.text_in(text)
        n_img=image.shape[1];n_total=n_img+text.shape[1]
        cos,sin=self.rope(t,h,w,text.shape[1],image.device,image.dtype)
        if self.text_trans is not None:
            text=self.text_refiner(text,text_mask,cos[n_img:],sin[n_img:])
        attention_mask=pack=None
        if text_mask is not None:
            valid_text=text_mask.bool()
            attention_mask=torch.cat((torch.ones((b,n_img),device=x.device,dtype=torch.bool),valid_text),1)[:,None,None,:]
            if _FA3:
                lens=n_img+valid_text.sum(-1)
                cu_seqlens=torch.zeros(b+1,device=x.device,dtype=torch.int32)
                cu_seqlens[1:]=lens.cumsum(0).to(torch.int32)
                cols=torch.arange(n_total,device=x.device)
                valid2d=cols[None,:]<lens[:,None]
                pack_idx=(torch.arange(b,device=x.device)[:,None]*n_total+cols[None,:])[valid2d]
                pos_idx=cols[None,:].expand(b,-1)[valid2d]
                bound=MAX_SEQLEN_IMAGE if t==1 else MAX_SEQLEN_VIDEO
                if n_total>bound:raise ValueError(f'n_total={n_total} exceeds FA3 bound={bound}')
                pack=(pack_idx,pos_idx,cu_seqlens,bound)
        hidden=stack_stream(torch.cat((image,text),1),self.n_streams)
        ts=timestep.float().reshape(-1).expand(b)
        freq=10000**(-torch.arange(128,device=x.device,dtype=torch.float32)/128)
        emb=torch.cat(((ts[:,None]*freq).cos(),(ts[:,None]*freq).sin()),-1)
        cond=self.time_in(emb.to(self.time_in[0].weight.dtype))
        if self.adaln_basis is not None:cond=self.adaln_basis(cond.to(self.adaln_basis.weight.dtype))
        self.moe_aux_losses=[]
        if self.has_moe:
            for block in self.blocks:block.last_moe_aux=None
        features=[] if feature_layers is not None else None
        use_ckpt=self.activation_checkpointing and self.training and torch.is_grad_enabled() and feature_layers is None
        from torch.utils.checkpoint import checkpoint
        for i,block in enumerate(self.blocks):
            if use_ckpt and i in self._ac_block_ids:
                hidden=checkpoint(block,hidden,cond,cos,sin,attention_mask,pack,n_img,
                                  denoising_step=self._moe_denoising_step,use_reentrant=False)
            else:
                hidden=block(hidden,cond,cos,sin,attention_mask,pack,n_img,denoising_step=self._moe_denoising_step)
            if block.last_moe_aux is not None:self.moe_aux_losses.append(block.last_moe_aux)
            if features is not None and i in feature_layers:features.append(read_out(hidden,self.readout))
        image=self.norm(read_out(hidden[:,:t*h*w],self.readout))
        shift,scale=self.head_mod(cond).to(image.dtype).chunk(2,-1)
        image=image*(1+scale).unsqueeze(1)+shift.unsqueeze(1)
        image=self.out(image).reshape(b,t,h,w,pt,ph,pw,self.out_channels)
        out=image.permute(0,7,1,4,2,5,3,6).reshape(b,self.out_channels,t*pt,h*ph,w*pw)
        return (out,features) if features is not None else out


def load_mhc_compatible_state_dict(model,state,metadata=None,allow_text_input_resize=True):
    """Strict dense->mHC migration; only new mHC tensors and text width may differ.

    Own mHC snapshots are strict: a partial snapshot or the old incompatible mHC
    topology must fail, rather than silently reinitializing its routing parameters.
    """
    current=model.state_dict()
    incoming_mhc=any(is_mhc_parameter_name(k) for k in state) or (isinstance(metadata,dict) and isinstance(metadata.get('mhc'),dict))
    if incoming_mhc:
        old=metadata.get('mhc') if isinstance(metadata,dict) else None
        expected=model.semantic_metadata()['mhc']
        if not isinstance(old,dict) or old.get('version')!=MHC_VERSION:
            raise ValueError('Incompatible/unknown legacy mHC checkpoint topology; dense source or v2 metadata required')
        for key in ('n_streams','sinkhorn_iters','mapping_fp32','input','readout','transfer_post'):
            if old.get(key)!=expected.get(key):raise ValueError(f'mHC checkpoint mismatch: {key}')
        missing=sorted(set(current)-set(state));unexpected=sorted(set(state)-set(current))
        mismatched=sorted(k for k in set(current)&set(state) if current[k].shape!=state[k].shape)
        if missing or unexpected or mismatched:
            raise ValueError(f'Incomplete/incompatible mHC checkpoint: missing={missing}, unexpected={unexpected}, shapes={mismatched}')
        model.load_state_dict(state,strict=True)
        model.last_migration_report={'source':'mhc','matched':len(state),'new_mhc':[],
                                     'reinitialized':[],'ignored':[]}
        return len(state),0,0
    new=sorted(k for k in current if is_mhc_parameter_name(k))
    matched={};resized=[];unexpected=sorted(set(state)-set(current))
    failures=[]
    for key,value in state.items():
        if key not in current:continue
        if value.shape==current[key].shape:matched[key]=value
        elif (allow_text_input_resize and key=='text_in.weight' and value.ndim==2
              and current[key].ndim==2 and value.shape[0]==current[key].shape[0]):
            resized.append(key)
        else:failures.append(key)
    missing_dense=sorted(set(current)-set(matched)-set(new)-set(resized))
    if unexpected or failures or missing_dense:
        raise ValueError(f'Unsafe dense->mHC migration: missing_dense={missing_dense}, unexpected={unexpected}, shape_mismatch={failures}')
    for block in model.blocks:
        block.mhc_attn.reset_mhc_parameters();block.mhc_ffn.reset_mhc_parameters()
    with torch.no_grad():model.readout.fill_(1/model.n_streams)
    result=model.load_state_dict(matched,strict=False)
    expected_missing=set(new)|set(resized)
    if set(result.missing_keys)!=expected_missing or result.unexpected_keys:
        raise RuntimeError('Internal mHC migration whitelist mismatch')
    model.last_migration_report={'source':'dense','matched':len(matched),'new_mhc':new,
                                 'reinitialized':resized,'ignored':resized,
                                 'note':'Only text input weight is resized; all remaining dense tensors are inherited.'}
    return len(matched),len(result.missing_keys),len(resized)
