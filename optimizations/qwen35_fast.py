"""Explicit, fail-closed Qwen3.5 kernel selection for the optimized entry point."""
from contextlib import contextmanager
import functools
import importlib.metadata
import inspect
import logging


def compatible_dispatch(function):
    """Match Transformers' dispatch policy: do not forward unrelated model kwargs."""
    accepted = set(inspect.signature(function).parameters)

    @functools.wraps(function)
    def dispatch(*args, **kwargs):
        return function(*args, **{key: value for key, value in kwargs.items() if key in accepted})

    dispatch.fast_implementation = function
    return dispatch


def resolve_kernels():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        from causal_conv1d import causal_conv1d_fn
    except Exception as exc:
        raise RuntimeError('Fast kernels unavailable. Activate gwp_qwen35_fast; refusing silent torch fallback.') from exc
    required = {'g', 'beta', 'initial_state', 'output_final_state', 'use_qk_l2norm_in_kernel'}
    if not required <= set(inspect.signature(chunk_gated_delta_rule).parameters):
        raise RuntimeError('FLA GDN API is incompatible with the installed Qwen3.5 implementation')
    if not hasattr(modeling, '_giga_reference_chunk'):
        modeling._giga_reference_chunk = inspect.unwrap(modeling.torch_chunk_gated_delta_rule)
        modeling._giga_reference_conv = inspect.unwrap(modeling.causal_conv1d_fn)
    return modeling, chunk_gated_delta_rule, causal_conv1d_fn


@contextmanager
def kernel_mode(mode):
    """Use one model/one set of inputs for reference versus fused comparisons."""
    modeling, chunk, convolution = resolve_kernels()
    old = modeling.torch_chunk_gated_delta_rule, modeling.causal_conv1d_fn
    if mode == 'fast':
        replacement = compatible_dispatch(chunk), compatible_dispatch(convolution)
    elif mode == 'reference':
        replacement = modeling._giga_reference_chunk, modeling._giga_reference_conv
    else:
        raise ValueError(mode)
    modeling.torch_chunk_gated_delta_rule, modeling.causal_conv1d_fn = replacement
    try:
        yield
    finally:
        modeling.torch_chunk_gated_delta_rule, modeling.causal_conv1d_fn = old


def final_features(encoder, inputs, layer=None):
    final = layer == -1 and getattr(encoder.config, 'model_type', None) == 'qwen3_5_text'
    out = encoder(**inputs, use_cache=False, return_dict=True,
                  output_hidden_states=layer is not None and not final)
    return out.last_hidden_state if final or layer is None else out.hidden_states[layer]


def kernel_self_test(device):
    """Validate both fused kernels before allowing a new training run to proceed."""
    import torch
    modeling, chunk, convolution = resolve_kernels()
    generator = torch.Generator(device=device).manual_seed(1729)
    dtype = torch.bfloat16
    random = lambda shape: torch.randn(shape, generator=generator, device=device, dtype=dtype)
    q, k, v = [random((2, 65, 32, 128)) for _ in range(3)]
    g = -torch.rand((2, 65, 32), generator=generator, device=device)
    beta = torch.rand((2, 65, 32), generator=generator, device=device, dtype=dtype)
    with torch.no_grad():
        reference, _ = modeling._giga_reference_chunk(
            q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
        result, _ = chunk(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        relative = (result.float()-reference.float()).square().mean().sqrt() / reference.float().square().mean().sqrt().clamp_min(1e-8)
        assert torch.isfinite(result).all() and relative < 0.02, f'GDN relative RMS error: {float(relative)}'
        # The encoder uses a channel-last strided view, not a contiguous [B,C,T] tensor.
        x, weight = random((2, 65, 128)).transpose(1, 2), random((128, 4))
        ref_conv = modeling._giga_reference_conv(x, weight, activation='silu')
        fast_conv = convolution(x, weight, activation='silu')
        conv_error = (fast_conv.float()-ref_conv.float()).square().mean().sqrt() / ref_conv.float().square().mean().sqrt().clamp_min(1e-8)
        assert torch.isfinite(fast_conv).all() and conv_error < 0.02, f'conv relative RMS error: {float(conv_error)}'
    return {'gdn_relative_rms': float(relative), 'conv_relative_rms': float(conv_error)}


def install_training_optimizations(training, device, self_test=True):
    modeling, chunk, convolution = resolve_kernels()
    test_result = kernel_self_test(device) if self_test else None
    modeling.torch_chunk_gated_delta_rule = compatible_dispatch(chunk)
    modeling.causal_conv1d_fn = compatible_dispatch(convolution)
    original_build, reference_encode = training.build_text_encoder, training.encode_text

    def checked_build(tc, dtype, device):
        import torch
        encoder = original_build(tc, dtype, device)
        tokenizer = training.build_tokenizer(tc)
        texts = training.chat_format_prompts(tokenizer,
            ['A realistic portrait of a woman wearing a red coat.', '白色桌子上并排摆放三个红苹果。'],
            tc.get('use_chat_template'), tc.get('enable_thinking', True))
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True,
                           max_length=tc.get('max_length', 2048)).to(device)
        with torch.no_grad():
            with kernel_mode('reference'):
                reference = reference_encode(encoder, inputs, tc.get('layer')).float()
            result = final_features(encoder, inputs, tc.get('layer')).float()
            mask = inputs.attention_mask.bool()
            error = (result[mask]-reference[mask]).square().mean().sqrt() / reference[mask].square().mean().sqrt().clamp_min(1e-8)
            if not torch.isfinite(result).all() or error >= 0.05:
                raise RuntimeError(f'Full encoder feature validation failed: relative RMS={float(error)}')
        logging.getLogger('giga_exp').info('FAST_ENCODER_FEATURES relative_rms=%g', float(error))
        print('FAST_ENCODER_FEATURES relative_rms=', float(error), flush=True)
        return encoder

    training.build_text_encoder = checked_build
    training.encode_text = final_features
    report = {'fla_core': importlib.metadata.version('fla-core'),
              'causal_conv1d': importlib.metadata.version('causal-conv1d'),
              'gdn_implementation': chunk.__module__, 'conv_implementation': convolution.__module__,
              'final_layer': -1, 'self_test': test_result}
    logging.getLogger('giga_exp').info('FAST_QWEN35 %s', report)
    print('FAST_QWEN35', report, flush=True)
    return report


def choose_weight_source(output_dir, bootstrap):
    from pathlib import Path
    import re
    own = Path(output_dir)/'checkpoints/weights'
    if any(re.fullmatch(r'step-\d+\.(safetensors|pt)', p.name) for p in own.glob('step-*')):
        return own
    return Path(bootstrap)


def install_resume_bootstrap(training, bootstrap):
    """Only the first optimized launch imports the source run's actual step/cursor."""
    from pathlib import Path
    import torch.distributed as dist
    original = training.load_pure_weights

    def load(engine, output_dir, checkpoint_path=None):
        if checkpoint_path is not None:
            return original(engine, output_dir, checkpoint_path)
        distributed = dist.is_available() and dist.is_initialized()
        selected = [None]
        if not distributed or dist.get_rank() == 0:
            selected[0] = str(choose_weight_source(output_dir, bootstrap))
        if distributed:
            dist.broadcast_object_list(selected, src=0)
        if Path(selected[0]) == Path(bootstrap):
            logging.getLogger('giga_exp').info('Bootstrap optimized run from %s; preserve source step, samples and cursor', selected[0])
        return original(engine, output_dir, selected[0])

    training.load_pure_weights = load
