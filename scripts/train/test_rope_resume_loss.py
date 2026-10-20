"""诊断: step-12000 权重在不同 rope_axes_dims 配置下的 forward loss。

copy5 (48卡) 在 step 12001-12082 的 loss≈0.436, copy6 (24卡) 恢复同一权重后 loss≈1.046。
权重 matched=139/139, 差异只能来自前向行为 (RoPE 配置)。哪个配置 loss 接近 0.44, copy5 就是用它训练的。
"""
import sys, yaml, torch
sys.path.insert(0, ".")
from safetensors.torch import load_file
from scripts.train.train_giga_exp import collate, encode_text, build_text_encoder, wan_latent_stats
from data.lmdb_dataset import GigaLMDBDataset
from models.giga_transformer import GigaExpTransformer

device = "cuda:0"
cfg = yaml.safe_load(open("configs/stage_1.yaml"))
mc = cfg["model"]
tc = cfg["components"]["text_encoder"]

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(tc["tokenizer_path"], local_files_only=True)
text_encoder = build_text_encoder(tc, torch.bfloat16, device)
vae = None
from diffusers import AutoencoderKLWan
vae = AutoencoderKLWan.from_pretrained(tc["wan_vae"]["pretrained_path"] if "wan_vae" in tc else cfg["components"]["wan_vae"]["pretrained_path"], subfolder="vae", torch_dtype=torch.float32, local_files_only=True).to(device)
vae.eval()
latents_mean, latents_std_inv = wan_latent_stats(vae, device)

data = GigaLMDBDataset(cfg["dataset"], seed=42, index_cache_dir="/tmp/rope_diag_cache")
bucket = data.bucket_indices.get("192_square") or next(iter(data.bucket_indices.values()))
records = [data.records[i] for i in bucket[:8]]
batch = collate([data[i] for i in bucket[:8]])
pixels = batch["pixel_values"].to(device)
print("captions:", [c[:30] for c in batch["captions"]])

with torch.no_grad():
    latents = (vae.encode(pixels.float()).latent_dist.sample() - latents_mean) * latents_std_inv
    prompts = [str(c).strip() or tokenizer.eos_token for c in batch["captions"]]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=tc.get("max_length", 512)).to(device)
    text, mask = encode_text(text_encoder, inputs, tc.get("layer")), inputs.attention_mask.bool()

state = load_file("/home/jovyan/pfs/users/zhanqian.wu/exp/giga_exp_image/stage_1/checkpoints/weights/step-00012000.safetensors")
torch.manual_seed(0)
sigma = torch.full((latents.shape[0],), 0.5, device=device)
noise = torch.randn_like(latents)
xt = (1 - sigma[:, None, None, None, None]) * latents + sigma[:, None, None, None, None] * noise

for dims in ([16, 48, 48, 16], [32, 32, 32, 32]):
    model = GigaExpTransformer(in_channels=mc["in_channels"], out_channels=mc["out_channels"], text_dim=mc["text_dim"],
                               dim=mc["dim"], ffn_dim=mc["ffn_dim"], heads=mc["heads"], layers=mc["layers"],
                               rope_theta=mc["rope_theta"], rope_axes_dims=dims).to(device=device, dtype=torch.bfloat16)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"dims={dims}: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    with torch.no_grad():
        pred = model(xt.to(torch.bfloat16), sigma * 1000, text, mask)
        loss = (pred.float() - (noise - latents).float()).square().mean().item()
    print(f"  dims={dims} -> loss={loss:.4f}  (copy5 参照 ≈0.44, copy6 参照 ≈1.05)")
    del model
    torch.cuda.empty_cache()
