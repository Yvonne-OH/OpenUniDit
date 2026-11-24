"""Offline preflight of the real frozen encoder and inherited production DiT."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'thirdparty')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
import yaml
from models.text_encoding import build_tokenizer, build_text_encoder, chat_format_prompts, encode_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/stage_1_5b_qwen35_4b_tar.yaml')
    parser.add_argument('--encoder-only', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    tc = cfg['components']['text_encoder']
    device, dtype = torch.device('cuda:0'), torch.bfloat16
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = build_tokenizer(tc)
    started = time.monotonic()
    encoder = build_text_encoder(tc, dtype, device)
    print('ENCODER_LOADED', type(encoder).__name__, 'seconds', round(time.monotonic()-started, 2), flush=True)
    assert not any(p.requires_grad for p in encoder.parameters())
    assert not hasattr(encoder, 'lm_head') and not hasattr(encoder, 'visual')
    texts = chat_format_prompts(tokenizer, ['一位穿红衣服的女孩，手里拿着三朵花。', 'A cat.', tokenizer.eos_token], True, tc['enable_thinking'])
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=tc['max_length']).to(device)
    with torch.no_grad():
        features = encode_text(encoder, inputs, tc['layer'])
        solo = tokenizer([texts[1]], return_tensors='pt').to(device)
        single = encode_text(encoder, solo, tc['layer'])
        length = int(inputs.attention_mask[1].sum())
        delta = (features[1, :length].float() - single[0].float()).square().mean().sqrt()
        scale = single.float().square().mean().sqrt().clamp_min(1e-8)
        rel_error = float(delta / scale)
        assert rel_error < 0.08, ('right padding changes valid tokens', rel_error)
        assert features.shape[-1] == cfg['model']['text_dim'] == 2560
        assert torch.isfinite(features).all()
        # Exercise the configured length limit and padding in the hybrid linear/full attention model.
        long_inputs = tokenizer([texts[0] * 200, texts[1]], return_tensors='pt', padding=True,
                                truncation=True, max_length=tc['max_length']).to(device)
        started = time.monotonic()
        long_features = encode_text(encoder, long_inputs, tc['layer'])
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        assert long_features.shape == (2, 2048, 2560)
        assert torch.isfinite(long_features).all()
    print('ENCODER_FORWARD_OK', tuple(long_features.shape), 'seconds', round(elapsed, 3),
          'padding_relative_error', rel_error, flush=True)
    # Inference must use exactly the same tokenizer/template/feature layer.
    from scripts.infer.infer_giga_exp import encode_cfg_prompts
    with torch.no_grad():
        pos, pmask, neg, nmask = encode_cfg_prompts(tokenizer, encoder, ['A cat.'], '', device,
            tc['layer'], tc['max_length'], use_chat_template=True, enable_thinking=tc['enable_thinking'])
    assert pos.shape[-1] == neg.shape[-1] == 2560 and pmask.any() and nmask.any()
    del long_features, long_inputs
    if not args.encoder_only:
        from scripts.train.train_giga_exp import load_init_weights, wan_latent_stats
        from models.giga_transformer import GigaExpTransformer
        from diffusers import AutoencoderKLWan
        model_cfg = dict(cfg['model'])
        model_cfg.pop('type', None)
        model = GigaExpTransformer(**model_cfg).to(device, dtype=dtype).train()
        checkpoint = load_init_weights(model, cfg['training']['init_from'], strict=True)
        dc = cfg['components']['wan_vae']
        vae = AutoencoderKLWan.from_pretrained(dc['pretrained_path'], subfolder=dc.get('subfolder', 'vae'),
                torch_dtype=dtype, local_files_only=True).eval().requires_grad_(False).to(device)
        mean, invstd = wan_latent_stats(vae, device)
        losses = []
        for size, batch in [(512, 1), (256, 2)]:
            with torch.no_grad():
                pixels = torch.randn(batch, 3, 1, size, size, device=device, dtype=dtype)
                latents = (vae.encode(pixels).latent_dist.sample().float() - mean) * invstd
                text = features[:batch]
                mask = inputs.attention_mask[:batch].bool()
                sigma = torch.rand(batch, device=device)
                sigma = 3 * sigma / (1 + 2 * sigma)
                noise = torch.randn_like(latents)
                xt = (1 - sigma[:,None,None,None,None]) * latents + sigma[:,None,None,None,None] * noise
            pred = model(xt.to(dtype), sigma * 1000, text, mask)
            loss = (pred.float() - (noise-latents)).square().mean()
            (loss / 2).backward()
            assert torch.isfinite(loss)
            assert model.text_in.weight.grad is not None and torch.isfinite(model.text_in.weight.grad).all()
            losses.append(float(loss.detach()))
            print('DIT_BACKWARD_OK', size, batch, losses[-1], flush=True)
        assert all(p.grad is None for p in encoder.parameters())
        print('STRICT_INHERITANCE_OK', checkpoint, 'parameters', len(model.state_dict()), flush=True)
        result = {'encoder_forward_seconds_b2_l2048': elapsed, 'padding_relative_error': rel_error,
                  'checkpoint': str(checkpoint), 'matched': len(model.state_dict()), 'losses': losses,
                  'peak_gpu_gib': torch.cuda.max_memory_allocated()/2**30}
        out = Path(cfg['output_dir']) / 'preflight'
        out.mkdir(exist_ok=True)
        (out / 'smoke_result.json').write_text(json.dumps(result, indent=2) + '\n')
        print('PREFLIGHT_OK', result, flush=True)


if __name__ == '__main__':
    main()
