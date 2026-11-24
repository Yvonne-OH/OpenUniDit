"""Frozen text conditioning shared by training, validation and inference."""
from transformers import AutoModel, AutoTokenizer


def build_tokenizer(tc):
    tokenizer = AutoTokenizer.from_pretrained(tc['tokenizer_path'], local_files_only=True)
    tokenizer.padding_side = tc.get('padding_side', 'right')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_text_encoder(tc, dtype, device):
    kind = tc.get('type')
    kwargs = dict(torch_dtype=dtype, local_files_only=True)
    if tc.get('attn_implementation'):
        kwargs['attn_implementation'] = tc['attn_implementation']
    if kind == 'qwen3_vl':
        from transformers import Qwen3VLForConditionalGeneration
        encoder = Qwen3VLForConditionalGeneration.from_pretrained(tc['pretrained_path'], **kwargs)
    elif kind == 'qwen3_5':
        from transformers import Qwen3_5ForConditionalGeneration
        # Load with the official checkpoint key mapping, then keep only the text
        # backbone. Do not allocate vision parameters on GPU or produce vocab logits.
        full_model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
            tc['pretrained_path'], output_loading_info=True, **kwargs)
        if loading.get('missing_keys') or loading.get('mismatched_keys'):
            raise RuntimeError(f'Incomplete Qwen3.5 checkpoint: {loading}')
        encoder = full_model.model.language_model
        del full_model
    else:
        encoder = AutoModel.from_pretrained(tc['pretrained_path'], **kwargs)
    encoder=encoder.eval().requires_grad_(False).to(device)
    encoder._giga_frozen_inference=bool(tc.get('frozen_inference_mode',False))
    if kind=='qwen3_5' and tc.get('reference_gdn_cuda_graph',False):
        from .reference_gdn_graph import ReferenceGDNGraphCache,install_reference_graph_dispatch
        install_reference_graph_dispatch()
        encoder._giga_reference_graph_cache=ReferenceGDNGraphCache(
            tc.get('gdn_graph_max_entries',8),tc.get('gdn_graph_max_chunks',8))
    return encoder


def encode_text(encoder, inputs, layer=None):
    import torch
    from contextlib import nullcontext
    final=layer==-1 and getattr(encoder.config,'model_type',None)=='qwen3_5_text'
    cache=getattr(encoder,'_giga_reference_graph_cache',None)
    token=None
    if cache is not None:
        from .reference_gdn_graph import activate,deactivate
        token=activate(cache)
    inference=bool(getattr(encoder,'_giga_frozen_inference',False))
    try:
        with torch.inference_mode() if inference else nullcontext():
            out=encoder(**inputs,use_cache=False,return_dict=True,
                        output_hidden_states=layer is not None and not final)
            hidden=out.last_hidden_state if final or layer is None else out.hidden_states[layer]
        # A training Linear saves its input for backward; inference tensors cannot be saved.
        return hidden.clone() if inference and not torch.is_inference_mode_enabled() else hidden
    finally:
        if token is not None:deactivate(token)


def chat_format_prompts(tokenizer, prompts, use_chat_template, enable_thinking=True):
    if not use_chat_template:
        return prompts
    return [tokenizer.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=enable_thinking)
            for p in prompts]


def text_encoder_metadata(tc):
    keys = ('type', 'pretrained_path', 'tokenizer_path', 'revision', 'layer', 'max_length')
    result = {key: tc.get(key) for key in keys}
    result.update(use_chat_template=bool(tc.get('use_chat_template')),
                  enable_thinking=bool(tc.get('enable_thinking', True)),
                  padding_side=tc.get('padding_side', 'right'))
    return result
