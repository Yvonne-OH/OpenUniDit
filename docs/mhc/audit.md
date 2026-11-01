# Giga-Exp 2-Stream mHC 改造 — Phase 0 代码库审计报告

> 本阶段仅读代码、未修改任何文件。恢复点：commit `e823805`（已推送 origin/main）。

## 1. 关键源码文件与类名

| 组件 | 文件 | 类/函数 |
| --- | --- | --- |
| DiT 主模型 | `models/giga_transformer.py` | `GigaExpTransformer`（L147） |
| Transformer Block | `models/giga_transformer.py` | `GigaBlock`（L86-144） |
| Attention | 同上（block 内联实现） | FA3 `flash_attn_func/varlen`，SDPA 回退（L108-141） |
| FFN | 同上 | `nn.Sequential(Linear, SiLU, Linear)`（L97） |
| Norm | `models/giga_transformer.py` | `RMSNorm`（L25）；QK per-head RMSNorm（L94-95） |
| AdaLN | 同上 | `self.mod = nn.Linear(dim, dim*6)`，产生 shift/scale/gate ×2（L98, L101） |
| RoPE | `models/giga_transformer.py` | `SpacetimeRoPE`（L40），4D axial (t,h,w,text_pos)，fp32 旋转（L123-126） |
| Output Head | 同上 | `self.norm`（RMSNorm）+ `self.out = Linear(dim, out_ch*patch_vol)`（L164-165, L245） |
| Pipeline | `pipeline/giga_pipeline.py` | `GigaExpPipeline`（from_pretrained 硬编码读 `configs/stage_1.yaml`） |
| 训练主脚本 | `scripts/train/train_giga_exp.py` | `main()`；`build_optimizer`、`set_lr`、`save_pure_weights` 等 |
| 训练启动 | `scripts/train/train_giga_exp_stage_1.sh` | torchrun + 环境注入 + compile cache |
| KD 训练（旁支） | `scripts/train/train_giga_exp_kd.py` | teacher = `models/giga_transformer_6b_teacher.py`，student 同 `GigaExpTransformer` |
| 配置 | `configs/stage_1.yaml`、`configs/zero1.json` | DeepSpeed ZeRO-1, bf16 |
| 数据 | `data/lmdb_dataset.py`、`data/samplers/bucket_sampler.py` | LMDB + bucket sampler（本次不改动） |

## 2. 实际模型配置（以代码 + checkpoint 为准，非任务书叙述值）

来自 `configs/stage_1.yaml` 的 `model:` 段与 checkpoint 内嵌 `giga_semantic_metadata`（两者一致）：

- `dim=1536, layers=28, heads=12 (head_dim=128), ffn_dim=4096`
- `text_dim=2560`（Qwen3-VL-4B-Instruct `hidden_states[-2]`，max_length 512）
- `in/out_channels=48`（Wan2.2 TI2V-5B VAE），`patch_size=[1,2,2]`（任务书叙述正确）
- `rope_axes_dims=[16,48,48,16]`, `theta=10000`
- **参数量实测 1020.56M（291 个张量），不是任务书叙述的 ~0.8B**。0.8B 是注释掉的 1792/20L 旧配置。
- 每 block 参数 ≈ 36.2M：qkv 7.08M + out 2.36M + FFN 12.58M + mod 14.17M + norms

### Checkpoint 现状（重要偏差）

任务书叙述"约 262k steps"。在 PFS 上**未找到 step-0026xxxx 权重文件**，实际可用：

| 文件 | step | 说明 |
| --- | --- | --- |
| `code/model_exp/checkpoints/giga_exp_image_stage_1_28layers/step-00200000.safetensors` | 200,000 | 纯权重 291 tensors，无 kd_projectors，元数据完整 |
| `exp/giga_exp_image/stage_1_28layers/checkpoints/weights/` | 最高 185,000 | 同结构 |
| `exp/giga_exp_image/stage_1_kd_28layers_v2/checkpoints/weights/step-00247500.safetensors` | 247,500 | KD 分支，多出 `kd_projectors.*` 键（推理可忽略） |

**迁移基准 checkpoint 需人工指定**（建议 200k 纯权重版；262k 是否存在于 BOS/其他机器待确认）。

## 3. 原始架构图与 forward 数据流

```
x [B,48,T,H,W] ──patchify[1,2,2]──► patch_in ─► img [B,N_img,1536]
text [B,L,2560] ──────────────────► text_in ─► txt [B,L,1536]
x = cat(img, txt)                                             # image 在前, text 在后 (顺序不可动)
ts ► cos/sin(128) ► time_in(256→1536→1536) ─► cond            # Flow Matching timestep 条件
cos,sin = SpacetimeRoPE(t,h,w,L)                              # fp32, image 先行网格 + text 追加

× 28 × GigaBlock(x, cond, cos, sin, mask, pack):
    shift_a,scale_a,gate_a, shift_f,scale_f,gate_f = mod(cond).chunk(6)   # AdaLN 调制源: cond, 不是 x
    y   = norm1(x)*(1+scale_a) + shift_a                                  # Pre-Norm AdaLN (attn)
    a   = FA3/dense-attn(RoPE(q_norm(y)), RoPE(k_norm(y)), v)             # qkv 无 bias
    x   = x + out(a) * gate_a                                             # ★ residual add #1 (gate 来自 mod)
    y   = norm2(x)*(1+scale_f) + shift_f                                  # Pre-Norm AdaLN (ffn)
    x   = x + ff(y) * gate_f                                              # ★ residual add #2 (SiLU-MLP)

image = out( norm(x[:, :N_img]) )                                             # 只取 image 前缀
out   = unpatchify ─► [B,48,T,H,W]                                            # v-prediction (noise - data)
```

## 4. 审计问题逐条回答

1. **每个 Block 几处 residual add？** 恰好 2 处（[giga_transformer.py L142](../../models/giga_transformer.py#L142) attn 后、[L144](../../models/giga_transformer.py#L144) ffn 后），无其他残差旁路。
2. **Attention 结构？** Pre-Norm：RMSNorm → AdaLN 调制 → QKV（无 bias）→ per-head QK-RMSNorm → RoPE(fp32) → FA3 dense / FA3 varlen（文本右 padding pack）/ SDPA 回退 → `out`。
3. **FFN 结构？** Pre-Norm：RMSNorm → AdaLN 调制 → `Linear(1536→4096) → SiLU → Linear(4096→1536)`，全无 bias。
4. **AdaLN 调制位置？** 调制参数由 `mod(cond)` 从 timestep 条件生成（6 组：attn 与 ffn 各 shift/scale/gate），作用于两条 Pre-Norm 分支；`gate_*` 是 residual gate，乘在子层输出上。
5. **residual scaling / gate？** 有 gate（AdaLN 风格 `x + out(a)*gate_a`），**无额外 residual scaling**。⚠️ mHC 接入时 gate 属于子层 F 的一部分，原样保留，不得重复加 gate。
6. **fused kernel？** FA3（含 varlen pack，`MAX_SEQLEN_IMAGE/VIDEO` host 常量分档）；可选 `torch.compile(dynamic=True)`（inductor 融合小 kernel）。无 Triton 自定义算子。
7. **checkpoint 参数命名规则？** `blocks.{0..27}.{norm1,norm2,qkv,q_norm,k_norm,out,ff.0,ff.2,mod}.{weight,bias}`（每 block 10 tensors）+ 顶层 `{patch_in,text_in,time_in.0,time_in.2,norm,out}`（11 tensors），共 291。safetensors 内嵌 `giga_semantic_metadata`（加载时校验不匹配仅 warning）。
8. **共享参数？** 无。RoPE `inv_freq` 是非持久 buffer（不入 state_dict）；无 tying。
9. **hidden state 特殊 reshape / packing？**
   - patchify/unpatchify 的 8 维 permute（[L205-206](../../models/giga_transformer.py#L205)、L246-247）；
   - FA3 varlen 的 `pack_idx/pos_idx/cu_seqlens`（有效前缀抽取，仅 attention 内部，pad 回填 0）；
   - image/text 拼接顺序：image 前缀在前，output head 按该前缀切片。**mHC 不得破坏此约定。**

## 5. 训练系统现状（影响迁移的设计）

- **优化器**：当前为纯 AdamW（muon 已 disabled），`build_optimizer` 按 `name.startswith("out.")` 与 ndim 分组；`set_lr` 手写 warmup+cosine，无独立 scheduler 对象。**mHC 新参数需要独立分组才能支持 `mhc_lr`**。
- **EMA**：无。
- **Gradient Checkpointing**：无（当前未启用）。
- **分布式**：DeepSpeed ZeRO-1 + bf16，grad clip 1.0（zero1.json），`torchrun` 多机约定。
- **checkpoint 双轨**：DeepSpeed 完整 state（`engine.save_checkpoint`）+ 纯权重快照（safetensors, `save_pure_weights`）；`load_compatible_state_dict` 按名称+shape 匹配、其余静默保留初始化值 —— **迁移脚本不能依赖它做严格校验，必须自行核对 missing/unexpected keys**。
- **resume**：`init_from`（optimizer 创建前载纯权重）+ `resume_mode` 三级策略，mHC 训练脚本可复用该路径加载转换后权重。
- **监控**：wandb/swanlab 双后端，已有 `train/loss_sigma_*` 分桶指标，可挂 `mhc/*` 指标。
- **validation**：每 1000 步 next-batch CFG 采样出图，mHC 侧可复用。

## 6. mHC 改造方案（Phase 0 产出，待批准后施工）

### 6.1 新增文件（全部独立，不触碰主分支文件）

```
models/giga_transformer_mhc.py      # MHCResidual + GigaBlockMHC + GigaExpTransformerMHC
models/mhc_residual.py              # 独立 mHC 模块 (MHCResidual, Sinkhorn, 映射)
pipeline/giga_pipeline_mhc.py       # 独立 pipeline
configs/mhc_2stream.yaml            # 显式 mhc.enabled: true (model 段复制 stage_1.yaml 并加 mhc 段)
scripts/train/train_giga_exp_mhc.sh # 独立启动脚本
tools/convert_checkpoint_to_mhc.py  # 转换脚本
tests/test_mhc_residual.py 等       # 测试
docs/mhc/*.md                       # 报告
```

### 6.2 接入点（只动 residual，其余原样）

- 模型级：embedding 输出 `x [B,N,C]` → `H = stack([x, zeros], dim=2) [B,N,2,C]`；block 循环改为传 `H`；output head 前加 read-out（初始 `[1,0]` 取 stream0，等价原始行为）。
- Block 级：`GigaBlockMHC` 内部把原两处 residual add 替换为两个**独立**的 `MHCResidual` 实例（`mhc_attn` / `mhc_ffn`）；原 norm1/norm2/qkv/q/k_norm/out/ff/mod 权重**名称与形状完全不变**（存于 `attn_core` / `ffn_core` 子模块或保持同名属性），保证转换脚本按原名搬运。
- 子层 F 的定义 = 原来的 `norm → adaLN → qkv/attn/out*gate` 与 `norm → adaLN → ff*gate`（gate 归属于 F，mHC 不重复加 gate）。
- F 内部是单流 `[B,N,C]`（pre 映射把 2 流混为 1 流喂给原 block），因此 **FA3/varlen/pack/RoPE/attention mask 逻辑零改动**。

### 6.3 数学布局（按任务书公式，显式 einsum）

```
H_in        [B,N,2,C]
H_pre, H_post [B,N,2]   (σ / 2σ, fp32)
H_res       [B,N,2,2]   (Sinkhorn, fp32)
pre_mixed   = einsum('bns,bnsc->bnc', H_pre,  H_in)        # 单流喂 F
H_out       = einsum('bnij,bnjc->bnic', H_res, H_in)
            + einsum('bns,bnc->bnsc', H_post, F(pre_mixed))
```

映射来源：`RMSNorm(vec(H)) [B,N,2C] → 独立 Linear(2C→8)` 拆为 pre(2)/post(2)/res(4) logits，全程 fp32，FP32/BF16 可配置。

### 6.4 Function-Preserving 初始化（迁移模式，与论文初始化分开实现）

- `H = [x, 0]`；目标 `H_pre≈[1,0], H_post≈[1,0], H_res≈I`，则 `H_out=[x+F(x), 0]`，逐 block 数学等价。
- 实现：映射 Linear 权重零初始化 + bias 置极端 logits（pre/post: `+8/-8` 量级；res: `c·I` 形式，Sinkhorn 对 `e^c·I` 一两轮即收敛到精确 I）。read-out 权重 `[1,0]`。
- 风险（如实报告）：sigmoid 饱和区梯度 ≈ 1e-4~3e-4，pre/post 映射初期学习慢；将在 Test D 专门验证 mapping 梯度非零，并提供 `init_eps`/logits 幅度可调。
- `init_mode: function_preserving`（迁移）与论文式动态映射初始化作为两个 `init_mode` 选项，不混用。

### 6.5 新增参数与显存预估

- 每 MHC 模块：Linear(2·1536→8) + bias ≈ 24.6k params；每 block 2 个 ≈ 49.2k；28 blocks ≈ **1.38M（+0.135%）**。
- 激活：每 block 多出 H 的 stream 维复制与若干 [B,N,2]/[B,N,2,2] 小张量；bf16 下每 token 每 block 约 +4·2·1536·2B ≈ 25KB 量级，具体以 Test E 实测为准。

### 6.6 主要风险清单

| 风险 | 对策 |
| --- | --- |
| `load_compatible_state_dict` 静默吞掉 missing keys | 转换脚本独立核对全部 291+新增 keys，产出 conversion_report.json |
| `torch.compile` 对 Sinkhorn 20 轮循环的处理 | 第一版 mHC 关闭 compile 或 eager 跑 mapping；对照 Test B 数值 |
| Muon/AdamW 分组按 ndim+前缀划分 | mHC 训练配置显式按参数名分组 `mhc_lr`；muon 保持 disabled |
| checkpoint 无 262k | 等人工指定基准（见 §2） |
| `giga_semantic_metadata` 不含 mhc 字段 | 转换脚本扩展 metadata（新增 `mhc` 段），旧字段保持兼容 |
| KD 旁支脚本 (`train_giga_exp_kd.py`) | 本次不改动；mHC 只提供独立训练脚本 |

## 7. 待人工确认事项

1. **迁移基准 checkpoint 用哪个？**（建议 `checkpoints/giga_exp_image_stage_1_28layers/step-00200000.safetensors`；若 262k 在 BOS/其他节点，请提供路径）
2. 6.1/6.2 的文件布局与接入点是否符合预期（独立 `giga_transformer_mhc.py` + 独立 pipeline/yaml/sh，主分支零改动）？
3. read-out 方案（stream0 直出，read-out 向量可学习）是否接受？
4. mHC 训练 yaml 的 output_dir / wandb 名称偏好？

**Phase 0 到此停止，未修改任何文件，等待人工确认后进入 Phase 1（独立 mHC 模块）。**
