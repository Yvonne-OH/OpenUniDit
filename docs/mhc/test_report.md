# mHC 测试报告（实际执行日志）

执行环境：CPU (torch 2.10.0+cu128)，日期 2026-11-01。所有数字为实际运行输出。

## Test A：Shape Test（tests/test_mhc_full_model.py）

覆盖 batch ∈ {1,2,3,4}，分辨率 (16,16)/(12,20)/(8,24)，text_len ∈ {1,5,7,11,33}，timestep ∈ [0,1000] 边界，T=2 视频档，n_streams ∈ {1,2,4}。全部通过，输出 shape 均为 `[B,C,T,H,W]`。

## Test B：Sinkhorn Test（tests/test_mhc_residual.py）

- 随机 logits (scale=1, 4×4, 20 轮)：row_err=8.89e-05, col_err=1.19e-07（列和经最后一次列归一化精确为 1）
- 100 轮：row_err=col_err=1.19e-07 < 1e-5 ✓
- 近恒等 logits（function-preserving 初始化的实际输入，diag=+12/offdiag=-12）：2 轮即精确 I
- **如实报告**：随机大幅值 logits（scale≥3）20 轮不收敛（row_err 2.4e-2~8.6e-2），收敛率随幅值变慢；mHC 实际场景 logits 由零初始化投影产生、幅值有界，训练监控指标 `mhc/*/sinkhorn_row_error` 会持续跟踪，若 row error 增大需调大 `sinkhorn_iters`
- 非负性、无 NaN/Inf、反向梯度有限 ✓

## Test C：Function Preservation（tests/test_function_preservation.py）

小配置（dim=64, 3 layers, n_streams=4），同输入对比原模型：

| 用例 | rel L2 | max abs | mean abs | cos |
| --- | --- | --- | --- | --- |
| fp32 seed0（含 varlen/mask 路径） | 1.13e-07 | 3.58e-07 | 4.29e-08 | 1.000 |
| fp32 seed1 | 1.08e-07 | 3.58e-07 | 4.04e-08 | 1.000 |
| fp32 seed2 | 1.23e-07 | 3.58e-07 | 4.64e-08 | 1.000 |
| dense 无 mask | 1.09e-07 | 3.58e-07 | 4.03e-08 | 1.000 |
| bf16 | 0.0 | 0.0 | 0.0 | 1.000 |

- FP32 rel L2 ≈ 1.1e-7，远优于 1e-4 目标 ✓
- BF16 完全一致（映射/Sinkhorn 走 fp32，误差低于 bf16 分辨率）✓
- 误差来源定位：唯一残差来源是 `2σ(-12)≈1.2e-5` 的 post 尾部泄漏与 `σ` 饱和（LP=8 时 rel=1.7e-4，已改 LP=12 降至 3.1e-6 模块级 / 1.1e-7 全模型级）
- 模块级测试（[tests/test_mhc_residual.py](../../tests/test_mhc_residual.py)）：stream0 相对误差=3.146e-06

## Test D：Gradient Test（tests/test_function_preservation.py）

- 全部参数 forward 后无 NaN/Inf ✓
- mHC mapping 梯度逐块非零：max|g|=1.635e-03, min|g|=7.383e-04（3 blocks）✓
- mapping.weight 梯度 max|g|=2.183e+01（模块级，含 res/Sinkhorn 路径）✓
- **饱和风险如实报告**：pre/post 分支 bias 初始位于 ±12 的 sigmoid 饱和区，对应坐标梯度 ≈1e-5 量级；res 分支经 log 域 Sinkhorn 归一化，梯度不受幅值影响，是主要可学习路径。训练期通过 `mhc/mapping_grad_norm`（fork 脚本中经 mhc_stats 记录）监控，若 pre/post 长期不动属预期，不代表 mHC 失效

## Test E：Full Model Test

见 Test A（本机 CPU 用小配置）；512×512 真实分辨率与多卡批量测试需在训练集群执行，归入 Phase 7 前置检查（init_from strict 加载 + warmup forward）。

## Test F：Checkpoint Round Trip（tests/test_checkpoint_conversion.py）

- 小模型 original → convert → save → reload(strict) → forward：rel_l2=1.090e-07 ✓
- 不完整 checkpoint（缺 blocks.1.*）被拒绝并报错 ✓
- mhc.enabled=false 的配置被拒绝 ✓
- **真实 280k checkpoint 转换**（tools/convert_checkpoint_to_mhc.py）：
  - 源：`outputs/bos_ckpt/giga_exp_image/stage_1_28layers/step-00280000.safetensors`（291 tensors, step=280000）
  - 输出：`/home/jovyan/pfs/users/zhanqian.wu/exp/giga_exp_image/stage_1_28layers_mhc/checkpoints/weights_mhc_converted/converted.safetensors`
  - 新增 113 张量，+8.259M params（+0.8093%）；unexpected=0；missing 全部为 mhc 新增键；无共享权重
  - 保存后回读逐张量 `torch.equal` 校验通过；conversion_report.json 已生成

## 结论

Phase 1–5 全部验收项通过。允许进入训练配置确认（Phase 7 短程训练需人工确认启动）。
