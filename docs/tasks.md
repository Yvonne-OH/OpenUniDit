# Giga-Exp 模型问题修复 — 任务清单

对应规范：[docs/spec.md](file:///home/jovyan/pfs/users/zhanqian.wu/code/model_exp/docs/spec.md)
状态：待审批

---

## T1 · 修正 `stage_1_1.yaml` 的 `resume_mode` → `auto`
- [ ] 改 [configs/stage_1_1.yaml](file:///home/jovyan/pfs/users/zhanqian.wu/code/model_exp/configs/stage_1_1.yaml#L143)：`resume_mode: /home/.../stage_1/checkpoints/weights/step-00170000.safetensors` → `resume_mode: auto`
- [ ] 确认 `stage_1p1/checkpoints/state/step-180000` 存在且完整（含 optimizer 分片）
- [ ] YAML 解析自检：`python -c "import yaml;print(yaml.safe_load(open('configs/stage_1_1.yaml'))['training']['resume_mode'])"`

## T2 · `gc.freeze()` + worker `gc.disable()`
- [ ] [data/datasets/image_database.py](file:///home/jovyan/pfs/users/zhanqian.wu/code/model_exp/data/datasets/image_database.py) `LmdbParquetDataset.__init__` 末尾：`import gc; gc.collect(); gc.freeze()`
- [ ] [scripts/train/train_giga_exp.py](file:///home/jovyan/pfs/users/zhanqian.wu/code/model_exp/scripts/train/train_giga_exp.py)：新增 `_worker_gc_disable(worker_id)` 函数
- [ ] 同文件 DataLoader 构造处挂 `worker_init_fn=_worker_gc_disable`（保持 `num_workers=4` / `prefetch_factor=4` / `persistent_workers=True` 不变）
- [ ] 自检：`python -c "import scripts.train.train_giga_exp"` 通过

## T3 · 解码模块运行期 hash 校验（防热编辑崩溃）
- [ ] [data/datasets/video_decode.py](file:///home/jovyan/pfs/users/zhanqian.wu/code/model_exp/data/datasets/video_decode.py)：新增 `_source_hash()` 计算本文件源码 hash
- [ ] `decode_isolated`：request 携带 `_expected_source_hash`
- [ ] `child_main`：启动时回传自身 hash，不一致 → 抛 `MediaExecutorBugError`（category=`executor_bug`，现有逻辑会快速失败）
- [ ] 自测：子进程运行期间改文件 → 断言抛 `executor_bug` 而不是 `TypeError`

## T4 · RoPE 修复后的训练侧验证（短跑，不长跑）
- [ ] 用 `stage_1p1/checkpoints/weights/step-00180000.safetensors` + 新 RoPE 在 512 桶推理一次
- [ ] 与 `stage_1p1/validation/step-00184000/rank0.png` 对比：结构连贯性、语义对齐
- [ ] 单机 8 卡冒烟 20~50 步：确认日志出现 `🔁 已恢复完整 DeepSpeed checkpoint step-180000`
- [ ] 确认无 `resize_frames() got an unexpected keyword`、无 `🚫 本 step 的全部 micro batch 均因坏数据跳过`
- [ ] 确认 worker RSS 无数量级跃升（`gc.freeze()` 生效）

## T5 · 集群 48 rank 重启验证
- [ ] 与用户确认可用节点（`stage_1` 192p 作业仍占用 6 机）
- [ ] 启动 `scripts/train/train_giga_exp_stage_1_1.sh`（多机），确认 `logs/<run_id>/node0..node5/` 各有独立 `console.log`
- [ ] 跑 500~1000 步，无 `TCPStore Broken pipe`
- [ ] 到 step=185000 后读 `validation/step-00185000/rank0.png`，判定语义是否恢复对齐

## T6 · 量化 `resize_long_edge` 补黑边比例（1% 采样）
- [ ] 抽 `stage_1p1` 训练桶 1% 样本，统计补黑像素占比 > 1% 的样本比例
- [ ] 若比例 > 10%：在 `resize_long_edge` / `cover` 间按源图比例自适应（需另行确认再改）

## T7 · 文档同步
- [ ] [docs/giga_exp_architecture.md](file:///home/jovyan/pfs/users/zhanqian.wu/code/model_exp/docs/giga_exp_architecture.md)：补 RoPE `i ↔ i+64` 配对约束
- [ ] 同上：补 `resize_long_edge` 补黑语义、跨 rank skip 的 all_reduce 同步协议
- [ ] 同上：补解码模块运行期禁止热编辑的约定

---

## 关键排期约束

⚠️ T3 会让**正在运行的 `stage_1`（192p）作业**在下一次解码时因 hash 不一致而失败
（它与 stage_1p1 共享 `video_decode.py`）。执行 T3 前需确认：
- (a) 等 `stage_1` 这一轮跑完，或
- (b) 用户接受 `stage_1` 被中断。

T1 / T2 / T4 不触碰 `video_decode.py` 与 `giga_transformer.py`，对 `stage_1` 无影响（`stage_1` 用的是 `configs/stage_1.yaml`，不受 `stage_1_1.yaml` 改动影响）。
