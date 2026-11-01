# mHC 架构与迁移说明

## 1. 实际修改范围

主分支文件 **零修改**（`models/giga_transformer.py`、`pipeline/giga_pipeline.py`、
`scripts/train/train_giga_exp.py`、`scripts/train/train_giga_exp_stage_1.sh`、`configs/stage_1.yaml` 均未动）。

新增文件：

| 文件 | 作用 |
| --- | --- |
| `models/mhc_residual.py` | 独立 mHC 模块：`MHCResidual` / log 域 Sinkhorn / stack_stream / read_out |
| `models/giga_transformer_mhc.py` | `GigaBlockMHC` + `GigaExpTransformerMHC`（子层核心逐行复制自 GigaBlock.forward 去掉 residual add） |
| `pipeline/giga_pipeline_mhc.py` | 独立推理 pipeline |
| `configs/mhc_2stream.yaml` | 显式 `mhc.enabled: true`，model 段与 stage_1.yaml 一致 + mHC 参数 |
| `tools/convert_checkpoint_to_mhc.py` | checkpoint 转换（strict 校验 + conversion_report.json） |
| `scripts/train/train_giga_exp_mhc.py` | 训练 fork：复用主分支组件，仅模型/优化器分组/日志不同 |
| `scripts/train/train_giga_exp_mhc.sh` | 独立启动脚本（MASTER_PORT 29511 与主分支错开） |
| `tests/test_*.py` ×4 | shape/sinkhorn/等价性/转换/全模型测试 |

## 2. 数据流（n_streams=4）

```
H = [x, 0, 0, 0]                                    # [B,N,4,C], x = patch_in/text_in 输出
× 28 × GigaBlockMHC:
    pre, post, res = mappings(RMSNorm(vec(H)))      # fp32, 投影 Linear(4C→24)
                                                    # pre/post [B,N,4], res=Sinkhorn→[B,N,4,4]
    s = Σ_k pre[k]·H[k]                             # 单流 [B,N,C] (einsum bns,bnsc->bnc)
    a = 原始 attention 子层(s, cond, cos, sin, mask, pack)   # 权重/逻辑与原模型完全一致
    H = res·H + post ⊗ a                            # (bnij,bnjc->bnic) + (bns,bnc->bnsc)
    s = Σ_k pre[k]·H[k]
    f = 原始 FFN 子层(s, cond)
    H = res·H + post ⊗ f
out = out_head(norm(readout·H[:, :N_img]))          # readout 初始 [1,0,0,0] = stream0 直出
```

- AdaLN gate (gate_a/gate_f) 归属于子层 F，mHC 不重复加 gate；
- 子层接收单流，FA3/varlen/pack/RoPE/mask 路径零改动；
- Attention 与 FFN 使用独立 `mhc_attn` / `mhc_ffn` 实例。

## 3. Function-Preserving 迁移初始化（与论文初始化分开的独立模式）

- 目标：`H_pre≈[1,0,0,0], H_post≈[1,0,0,0], H_res≈I, H=[x,0,0,0]` ⇒ `H_out=[x+F(x),0,0,0]`
- 实现：投影权重全零 + bias 极端 logits（pre: [+12,-12×3]；post: [0,-12×3]；res: 12·I 结构）；readout=[1,0,0,0]
- 实测：模块级 rel=3.1e-6，全模型 fp32 rel=1.1e-7，bf16 逐位一致
- **已知代价（如实报告）**：pre/post bias 处于 sigmoid 饱和区，对应坐标梯度 ≈1e-5；res 路径（log 域 Sinkhorn）梯度正常，是初期主要学习路径
- 论文式动态映射初始化保留为 `MHCResidual.init_paper_mode()`，与迁移初始化互斥

## 4. Checkpoint 映射

- 原 291 张量名 1:1 保留（`blocks.{i}.{norm1,norm2,qkv,q_norm,k_norm,out,ff.0,ff.2,mod}` + 顶层 11 个）
- 新增 113 张量：`blocks.{i}.{mhc_attn,mhc_ffn}.mapping.{weight,bias}`（112 个）+ `readout`（1 个）
- +8.259M params（1020.56M → 1028.82M，+0.809%）
- optimizer state 不迁移（第一版仅权重）；mHC 新参数 optimizer 状态全新，`mhc_lr` 独立分组
- `giga_semantic_metadata` 扩展 `mhc` 段（n_streams/init_mode/converted_from/source_step）

## 5. 已知限制与风险

1. `torch.compile` 在 mHC 训练中显式禁用（Sinkhorn 20 轮循环 eager 保证数值正确）
2. muon 优化器与 mHC 分组不兼容，第一版仅 adamw（脚本中显式 raise）
3. Sinkhorn 行和在 20 轮时对大幅值 logits 误差 ~1e-2（列和精确）；监控 `sinkhorn_row_error`，必要时调大 `sinkhorn_iters`
4. 真实 512×512 分辨率 / 多卡 / 短程训练的验证需在训练集群执行
