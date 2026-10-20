# FLUX.3 Image Klein Base 推理

使用项目内 vendored `diffusers/` 和本地 checkpoint 运行 FLUX.3 Image Klein Base 的 Diffusers modular pipeline。

## 目录

```text
model_exp/
├── diffusers/                         # Hugging Face Diffusers main
├── infer.py                           # 文生图/参考图编辑入口
├── scripts/infer/infer_flux3_klein_base.sh
├── outputs/
└── requirements.txt
```

## 环境

```bash
cd /home/jovyan/pfs/users/zhanqian.wu/code/model_exp
pip install -r requirements.txt
```

默认 checkpoint：

```text
/home/jovyan/pfs/users/zhanqian.wu/ckpt/flux-3-image-klein-base
```

checkpoint 下载完成后应至少包含 transformer 的两个分片和 text encoder 的两个分片。`infer.py` 会在加载模型前检查索引中声明的分片，并明确报告缺失文件。

## 文生图

```bash
CUDA_VISIBLE_DEVICES=0 \
PROMPT='A detailed cinematic photograph of a red fox in a snowy forest at golden hour.' \
OUTPUT=outputs/fox.png \
bash scripts/infer/infer_flux3_klein_base.sh
```

也可以直接调用：

```bash
python infer.py 'A detailed full-scene caption...' \
  --output outputs/fox.png \
  --seed 0 \
  --steps 50 \
  --guidance-scale 3.5
```

## 参考图编辑

```bash
python infer.py 'A complete caption describing the edited target image...' \
  --image input.png \
  --output outputs/edited.png
```

`--image` 最多可重复 10 次。

## 显存与 prompt upsampler

- Base 模型默认参数为 50 步、CFG 3.5，VAE 使用 FP32，其余组件使用 BF16。
- 显存不足时增加 `--cpu-offload`。
- 默认不加载独立的 9B prompt upsampler，因此应传入包含主体、构图、光照、材质和背景的完整场景描述。
- 已下载 PUP 或允许联网下载时，可增加 `--use-prompt-upsampler`；若需在线补文件，再增加 `--allow-download` 并设置 `HF_HUB_OFFLINE=0`。

## MOSS-Image-SFT 推理

仓库新增 [infer_moss_image.py](file:///home/jovyan/pfs/users/zhanqian.wu/code/model_exp/infer_moss_image.py)，默认使用本地 checkpoint `/home/jovyan/pfs/users/zhanqian.wu/ckpt/MOSS-Image-SFT`，并复用 checkpoint 中的 `pipeline_moss_image.py`。需要 PyTorch 2.9+、Diffusers 0.38+、Transformers 5.3+ 和 CUDA GPU。

```bash
cd /home/jovyan/pfs/users/zhanqian.wu/code/model_exp
CUDA_VISIBLE_DEVICES=0 python infer_moss_image.py \
  'A wide view of a refrigerator beside a kitchen counter with a fruit bowl. A small sticky note on the refrigerator door is handwritten in Chinese: "记得买牛奶和鸡蛋！"' \
  --output outputs/refrigerator.png \
  --seed 42
```

也可以使用脚本：

```bash
PROMPT='A red fox in a snowy forest at golden hour.' \
OUTPUT=outputs/fox.png \
bash scripts/infer/infer_moss_image_sft.sh
```

支持 JSON layout prompt、`--negative-prompt`、`--num-images`、`--cpu-offload` 和 `--uncond-skip-layers`。传入 `--uncond-skip-layers` 但不提供数字即可关闭默认的第 22 层 negative-branch skip。
