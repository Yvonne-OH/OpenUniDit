# benchmark/ — 文生图模型评测框架

对训练产出的 ckpt 做系统化评测：**批量生成 → 打分 → 汇总报告**。

```
benchmark/
├── prompts/                       # 评测 prompt 集 (jsonl, 由 build_prompts.py 生成)
│   ├── build_prompts.py           #   重建脚本 (源 URL 与采样逻辑)
│   ├── geneval_prompts.jsonl      #   GenEval 官方 553 条
│   ├── heim_prompts.jsonl         #   HEIM 场景子集 300 条 (PartiPrompts 分层采样)
│   ├── hpsv2_prompts.jsonl        #   HPS v2 官方 3200 条
│   └── qwen_image_bench_prompts.jsonl  # Qwen-Image 维度构造 100 条 (非官方)
├── scorers/
│   ├── base.py                    # 统一接口: score(images, prompts) -> list[float]
│   ├── aesthetic.py               # LAION aesthetic predictor v2 (sac_logos_v2)
│   ├── image_reward.py            # ImageReward v1.0 (vendored BLIP)
│   ├── hpsv2.py                   # HPS v2 (open_clip ViT-H-14 + 官方 ckpt)
│   ├── vlm_judge.py               # 本地 Qwen3-VL 判定器 (geneval 判定 / 多维打分)
│   └── _vendor/image_reward_blip/ # THUDM/ImageReward 的 BLIP 模块 (已适配 transformers 5.x)
├── generate.py                    # 批量生成 (复用 scripts/infer/infer_giga_exp.py 管线)
└── run.py                         # 打分 + report.json / report.md
```

统一 jsonl 格式：`{"id": str, "benchmark": str, "prompt": str, "meta": {...}}`

## 1. prompt 集说明与出处

| 文件 | 条数 | 来源 | 备注 |
|---|---|---|---|
| geneval_prompts.jsonl | **553** | [djghosh13/geneval](https://github.com/djghosh13/geneval) `prompts/evaluation_metadata.jsonl` | GenEval 官方仓库（任务描述中的 `zfj3214/GenEval` 为已失效镜像）。meta 保留原始 `tag`/`include`（对象类别、计数、颜色） |
| heim_prompts.jsonl | 300 | HEIM (NeurIPS 2023, [arXiv 2311.04287](https://arxiv.org/abs/2311.04287)) 所用场景数据源 **PartiPrompts** ([google-research/parti](https://github.com/google-research/parti) `PartiPrompts.tsv`, 全量 1632 条) 按 `Category` 分层采样 300 条（种子 42，可复现） | **截取方式**：HEIM 官方数据仓库 `stanford-crfm/heim` 已失效（404），故按论文所述场景取其底层官方数据集构建代表性子集；meta 记录 source/category/challenge |
| hpsv2_prompts.jsonl | **3200** | HF 数据集 [zhwang/HPDv2](https://huggingface.co/datasets/zhwang/HPDv2) `benchmark/{anime,concept-art,paintings,photo}.json` (4 类 × 800) | HPS v2 官方 benchmark prompts（与 pip 包 `hpsv2` 的 `utils.hps_prompt_files` 一致）；meta.category 标注类别 |
| qwen_image_bench_prompts.jsonl | 100 | **非官方**，按 Qwen-Image 技术报告 ([arXiv 2508.02324](https://arxiv.org/abs/2508.02324)) 评测维度构造：英文/中文文字渲染、多人交互、人物肖像、动物、食物、风景、建筑、艺术风格、科幻产品，10 维 × 10 条 | QwenLM/Qwen-Image 官方仓库未随附 benchmark prompts（已核对全部文件）；meta.dimension 标注维度 |

重建：`python benchmark/prompts/build_prompts.py --src-dir <已下载的源目录>`（或 `--download` 现场拉取）。

## 2. ckpt 目录约定

所有评测权重放在 `~/ckpt/`（即仓库外 `/home/jovyan/pfs/users/zhanqian.wu/ckpt/`，
可用环境变量 `BENCH_CKPT_ROOT` 覆盖）：

```
ckpt/
├── imagereward/
│   ├── ImageReward.pt              # ~1.8GB, 来源 hf-mirror.com/zai-org/ImageReward
│   └── med_config.json
├── hpsv2/
│   └── HPS_v2_compressed.pt        # ~1.97GB (v2.0), 来源 hf-mirror.com/spaces/xswu/HPSv2
├── aesthetic/
│   └── sac+logos+ava1-l14-linearMSE.pth   # ~3.6MB, 即社区所称 sac_logos_v2 (LAION aesthetic v2)
├── clip-vit-large-patch14/         # OpenAI CLIP ViT-L/14 (aesthetic 的底座), HF 格式
└── bert-base-uncased/              # ImageReward BLIP 文本编码器 tokenizer (vocab.txt 等)
```

VLM judge 使用 configs/stage_1.yaml `components.text_encoder.pretrained_path`
（Qwen3-VL-4B-Instruct，local_files_only），可用 `BENCH_VLM_PATH` 覆盖。

## 3. scorer 一览

| scorer | 分数量纲 | 依赖 | 说明 |
|---|---|---|---|
| `aesthetic` | 1~10 美学分 | transformers CLIP + torch | CLIP ViT-L 图像 embedding 归一化 → MLP 头 |
| `image_reward` | 归一化偏好分 (~N(0,1))，越高越好 | transformers + torch | 官方 pip 包与 transformers 5.x 冲突，故 vendor 其 BLIP 模块（`_vendor/image_reward_blip/`，改动仅限 file_utils import、post_init、get_head_mask、tokenizer 本地化四处兼容性修复），加载/打分逻辑按官方 `ImageReward.py` 等价重写 |
| `hpsv2` | 原始 logits（约 0~0.35），越高越好 | `open_clip_torch`（`pip install --user open_clip_torch`） | 官方 compressed ckpt 内含 CLIP ViT-H/14 全部权重 + 偏好头，`pretrained=None` 仅构建结构后整体覆盖 |
| `vlm_judge`(mode=geneval) | 每图 0~1 通过率 | 本地 Qwen3-VL-4B | 按 GenEval meta 生成判定清单（对象存在/计数/颜色/位置），VLM 输出 JSON verdict |
| `vlm_judge`(mode=score) | 1~10 均分 | 本地 Qwen3-VL-4B | alignment / aesthetics / detail 三维打分，score() 取均值，score_detailed() 给各维 |

所有 scorer 均支持 CPU（HPS/IR 首次加载较慢；VLM judge CPU 上单图约 1~3 分钟）。

## 4. 用法

### 4.1 批量生成（需 GPU）

```bash
python -m benchmark.generate \
    --bench hpsv2 \
    --checkpoint outputs/stage_1/checkpoints/weights/step-025000.safetensors \
    --num-seeds 4 --limit 100 \
    --height 128 --width 192 --steps 40 --guidance-scale 3.0 \
    --batch-size 8 --device cuda \
    --output-root outputs/bench
```

输出：`outputs/bench/<bench>/<ckpt_tag>/<id>_seed{seed}.png` + `manifest.json`。
复用方式：`scripts/` 不是 python 包，`generate.py` 用 importlib 按文件路径加载
`scripts/infer/infer_giga_exp.py`，复用其 ckpt 加载（含 AdaLN basis 迁移）、Qwen3-VL
编码、flow ODE `sample_batch()`、VAE 解码；不走 subprocess（infer CLI 不支持携带 id/meta 的 prompt）。

### 4.2 打分与报告（CPU 即可）

```bash
# 默认评分方式: geneval→vlm_geneval, heim/qwen_image_bench→vlm_score, hpsv2→hpsv2
python -m benchmark.run --bench all \
    --images-dir outputs/bench \
    --out outputs/bench/report.json --md outputs/bench/report.md

# 只评一个 bench / 覆盖 scorer (冒烟常用)
python -m benchmark.run --bench hpsv2 --images-dir /tmp/bench_smoke \
    --ckpt-tag smoke --scorer aesthetic --out /tmp/bench_smoke/report_smoke.json
```

报告内容：
- `report.json`：总览 + 每 benchmark 的 mean / 各维度正确率（geneval: existence/count/color/position）
  / 各维均分（heim、qwen_image_bench: alignment/aesthetics/detail）+ per_image 明细
- `report.md`：markdown 表格（总览表 + 分 bench 明细表）

`--limit N` 可限制每个 ckpt_tag 最多评 N 张图（冒烟）；同 id 多 seed 自动取 prompt 级均值。

## 5. GPU 占用提示

⚠️ 本框架的生成阶段需要 GPU（DiT 40 步 ODE 采样 + VAE 解码），**打分阶段全部支持 CPU**。
当前集群 8 卡被另一任务占满时：
- 可以正常做 prompts 构建、scorer 打分、报告生成（CPU）；
- 不要启动 `benchmark/generate.py`（会抢 GPU 显存）；等打标任务结束后再批量生成。
- VLM judge 在 CPU 上可用但慢（Qwen3-VL-4B bf16 ≈ 9GB 内存，`--device cpu`）。

## 6. 冒烟测试记录（全部 CPU，2026-11-08）

测试图片：`~/exp/giga_exp_image/edit_t2i_caption_sample/*.jpg`（3 张）

| 测试 | 命令要点 | 结果 |
|---|---|---|
| import 验证 | `from benchmark.scorers import ...; import benchmark.generate, benchmark.run` | 通过 |
| aesthetic | `AestheticScorer().score(3 imgs)` | `[5.7944, 5.6835, 5.5496]` |
| image_reward | `ImageRewardScorer().score(3 imgs, 3 prompts)` | `[-0.6955, -0.1584, 0.0834]` |
| vlm judge (geneval) | `judge_geneval(img, "a bench...", meta)` | 输出 checks（existence 未过: "The image does not contain a bench"）✓ |
| vlm judge (score) | `judge_score(img, prompt)` | `{'alignment': 1.0, 'aesthetics': 5.0, 'detail': 8.0}` ✓ |
| run.py 报告 | `--bench hpsv2 --images-dir /tmp/bench_smoke --scorer aesthetic` | report.json + report.md 生成 ✓ |
| hpsv2 | `HPSv2Scorer().score(3 imgs, 3 prompts)` (CPU, ViT-H-14) | `[0.2043, 0.2019, 0.1903]` ✓ |
| run.py 默认 hpsv2 | `--bench hpsv2 --images-dir /tmp/bench_smoke`（prompt 从 jsonl 按 id 回查） | report.md: hpsv2 mean 0.1988 ✓ |

## 7. 已知限制

- `hpsv2` 权重 `HPS_v2_compressed.pt` 来自 hf-mirror，直连 huggingface.co 不通；网络波动时用
  `curl -C - -o ... "https://hf-mirror.com/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt"` 续传。
- HEIM 子集为 PartiPrompts 分层采样（HEIM 官方 repo 已失效），与官方 HEIM leaderboard 数值不可直接对比；
  如需官方全量，待 `stanford-crfm/heim` 恢复后替换 `build_prompts.py` 的 build_heim。
- vlm judge 的 GenEval 判定是 VLM 版协议（官方 GenEval 用 object detection + 分割），
  数值口径为 "VLM 判定通过率"，适合模型间横向对比而非复刻官方分数。
