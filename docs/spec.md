# Giga-Exp 图像模型 — 问题排查结论与修复规范

生成日期：2026-10-22
适用仓库：`/home/jovyan/pfs/users/zhanqian.wu/code/model_exp`
适用实验：`stage_1p1`（512p）、`stage_1`（192p）

---

## 0. 一句话结论

上一轮提出的两项修复**只完成了一半**（launcher 分节点日志目录 ✅ / `gc.freeze()` ❌）；
本轮排查又新发现 **3 个问题**，其中 **RoPE 配对错位** 是影响最大的历史遗留 bug，
它很可能就是「512p 后 valid 语义对不上」的直接原因，而不只是「再适应期」。
当前**没有 stage_1p1 训练进程在跑**（该作业已于 11:18:24 崩溃），必须重启。

---

## 1. 已确认的问题清单（含证据）

### P1 · 多机作业崩溃根因 = 运行中热编辑导致的解码子进程代码不一致 【代码已在磁盘修复，未在集群验证】

- 证据：`exp/giga_exp_image/stage_1p1/logs/multi_zkey-0p8b-pretrain-stage1p1-copy1-master-0_23456_n6_stage_1_1/console.log`
  - `TypeError: resize_frames() got an unexpected keyword argument 'style'` × **1477**
  - `MediaCandidatesExhausted` × 546 → `🚫 本 step 的全部 micro batch 均因坏数据跳过` × 266
  - Root Cause: `rank 18 (worker-1 local_rank 2) exitcode 1` @ `11:18:24` → 其余 rank 全部 `TCPStore sendBytes failed: Broken pipe` 连锁
- 机理：`data/datasets/video_decode.py` 的解码子进程是**每次请求都 `subprocess` 重新读磁盘源码**的（`decode_isolated` → `Popen([sys.executable, __file__])`）。
  训练进程 08:38 启动时 import 的是旧版 `resize_frames`（无 `style` 形参）；
  11:17 之后磁盘文件被热编辑成「`read_images` 已透传 `style=style` / `resize_frames` 签名尚未更新」的半新半旧状态（文件最终 mtime 11:18:22），
  新 spawn 的子进程即命中 TypeError → 坏数据 → 整批跳过 → rank18 退出 → 48 rank 连锁崩溃。
- 现状：磁盘上 `read_images(path, width, height, condition_path=None, style="cover")` 与
  `resize_frames(frames, out_w, out_h, src_w, src_h, style="cover")` 两处签名已一致，
  实测 `style='cover'` / `style='resize_long_edge'` 均正常返回 `(3,1,512,384) uint8`。
- **残留风险**：只要在作业运行期间再编辑 `video_decode.py`（或它 import 的任何模块），同样的崩溃会重演。

### P2 · RoPE 4 轴配对错位 【已改代码，未做训练侧验证 —— 最高优先级】

- 证据（数值验证脚本）：旧实现 `cos[i] == cos[i+64]` → **False**；新实现 → **True**
- 机理：`models/giga_transformer.py` 的 `rotate_half` 用 `chunk(2)` 配对（`i ↔ i+64`），
  标准 RoPE 要求 `cos[i] == cos[i+64]`。旧 `SpacetimeRoPE.forward` 却按轴各自复制：
  ```python
  pieces.append(torch.cat((phase, phase), -1))   # 每轴 freq×2
  ```
  拼出来是 `[t×2, h×2, w×2, l×2]` 的分块重复 → `i` 与 `i+64` 落在**不同轴、不同频率**上，
  旋转不再是标准 2D 旋转，相对位置性质被破坏。
- 修复（已落盘）：
  ```python
  phase = torch.cat(pieces, -1)          # [N, head_dim/2] 每轴唯一频率
  phase = torch.cat((phase, phase), -1)  # i ↔ i+half 同频配对, 与 rotate_half 的 chunk(2) 对齐
  ```
  同时把旋转改为 fp32 计算再 cast 回 bf16（避免 bf16 cos/sin 相位量化）。
- **关键推论**：`stage_1` 从第 0 步到 178k、`stage_1p1` 到 184k **全程都是在错误 RoPE 下训练的**。
  RoPE 无参数（`inv_freq` 为 `persistent=False` buffer），所以 139 个张量照常 `matched`，但
  **权重的位置编码语义已改变** → 换回正确 RoPE 后需要一次重新适应，loss 会跳升。
  这也解释了为什么 512p 阶段「结构连贯但语义对不上」：位置外推正是依赖 RoPE 的相对性，
  而旧 RoPE 恰恰把这个性质破坏了。

### P3 · `stage_1_1.yaml` 的 `resume_mode` 指向了错误实验的 checkpoint 【未修】

- 现状（[configs/stage_1_1.yaml](file:///home/jovyan/pfs/users/zhanqian.wu/code/model_exp/configs/stage_1_1.yaml#L143)）：
  ```yaml
  resume_mode: /home/jovyan/pfs/users/zhanqian.wu/exp/giga_exp_image/stage_1/checkpoints/weights/step-00170000.safetensors
  ```
- 事实：`stage_1p1` 自己已有 `checkpoints/weights/step-00180000.safetensors`（10:37）
  **和完整 DeepSpeed state `checkpoints/state/step-180000`**（含 optimizer + lr_scheduler）。
  硬编码指向 `stage_1`（192p 实验）的 170000，会**丢掉 stage_1p1 已训的 512p 全部进展**。
- 附带问题：`resume_mode` 为显式路径时 `resume_tag = None`，代码不会走 `engine.load_checkpoint`，
  于是**永远拿不到 optimizer 状态**，每次重启都退化成「纯权重 + optimizer 重置」。

### P4 · `gc.freeze()` / worker 内 `gc.disable()` 未实现 【上一轮提出，未做】

- 全库 grep：`gc.freeze` / `gc.disable` / `import gc` **零命中**。
- 背景：每 rank 独立反序列化 **21.38 GiB** 索引 pkl（磁盘文件 22.96 GB），
  48 rank → 每节点 8 rank ≈ 171 GiB 常驻索引；再 fork 4 个 DataLoader worker，GC 触碰对象页会引发 COW 复制。
- 事实修正：copy1 那一轮（同样 48 rank）**内存是够的**——它从 08:38 跑到 11:18 训了约 2.5 小时（184000 → 184484）。
  所以 P4 不是崩溃根因（首轮 n6 run 在 08:35 的 master 消失也已被 P1 解释），但仍是低成本保险。

### P5 · launcher 分节点日志目录 【✅ 已完成】

- [scripts/train/train_giga_exp_stage_1_1.sh](file:///home/jovyan/pfs/users/zhanqian.wu/code/model_exp/scripts/train/train_giga_exp_stage_1_1.sh#L42-L47) 已按 `node${RANK}` 分目录。
- 注意：**崩溃的 copy1 run 用的是旧逻辑**，所以它的 `command.txt` 被 node5 覆盖、日志仍然混流（45 MB 单文件）；新 run 才会分目录。

### P6 · `resize_long_edge` 补黑边的训练分布未量化 【待查】

- `resize_long_edge` 在短边居中补黑（`video_decode.resize_frames`）。val 图未见黑边，
  但训练集里长宽比与桶比例差距大的样本会带黑边，模型可能学到黑边先验。
- 桶比例覆盖 1:1 / 4:3 / 3:2，且索引期按 `_nearest_bucket` 分配，理论上补黑边比例不高，但**未实测**。

---

## 2. 当前运行状态（2026-10-22 14:15 实测）

| 实验 | 分辨率 | 状态 | 最后步数 | 最后权重 |
|---|---|---|---|---|
| `stage_1p1` | 512p | **已崩溃**（11:18:24），无进程在跑 | 184484（validation 到 184000） | `stage_1p1/checkpoints/weights/step-00180000.safetensors` + 完整 state `step-180000` |
| `stage_1` | 192p | **仍在跑**（`multi_...-stage1-1022-...`，14:15 时 step=178015） | 178015 | `stage_1/checkpoints/weights/step-00175000.safetensors` |

- 本机 8 卡被 `rigel` 占用（6.4 GB / 100% util），stage_1p1 重启需走集群。
- 集群侧可用节点需与用户确认（stage_1 正在占 6 机）。

---

## 3. 修复方案（决策完备）

### F1 · 修正 `resume_mode` → `auto` 【必须】
- 文件：[configs/stage_1_1.yaml](file:///home/jovyan/pfs/users/zhanqian.wu/code/model_exp/configs/stage_1_1.yaml#L143)
- 改动：`resume_mode: auto`
- 生效路径：`auto` → `latest_state_tag(checkpoints/state)` → `step-180000` → `engine.load_checkpoint(..., load_optimizer_states=True, load_lr_scheduler_states=True)`
  → **从 step=180000 恢复，optimizer/momentum/lr_scheduler 全部续上**。
- 兜底：完整恢复失败时代码自动降级 `load_pure_weights(engine, output_dir)` → 取 `weights/step-00180000.safetensors` → step 仍保留 180000。
- 不再需要 `resume_mode` 指向 `stage_1` 的 170000。

### F2 · 索引加载后 `gc.freeze()` + DataLoader worker `gc.disable()` 【必须】
- 文件 A：[data/datasets/image_database.py](file:///home/jovyan/pfs/users/zhanqian.wu/code/model_exp/data/datasets/image_database.py) `LmdbParquetDataset.__init__` 末尾
  （records 已加载、分桶已完成之后）追加：
  ```python
  import gc
  gc.collect()
  gc.freeze()   # 索引对象移出 GC 追踪, fork 后不再 COW
  ```
- 文件 B：[scripts/train/train_giga_exp.py](file:///home/jovyan/pfs/users/zhanqian.wu/code/model_exp/scripts/train/train_giga_exp.py#L680-L682)
  新增 worker 初始化钩子并挂到 DataLoader：
  ```python
  def _worker_gc_disable(_worker_id):
      import gc
      gc.disable()   # worker 内不做周期 GC, 避免触碰父进程冻结页触发 COW
  ...
  loader = torch.utils.data.DataLoader(data, batch_sampler=batch_sampler,
                                       num_workers=cfg["training"].get("num_workers", 4),
                                       prefetch_factor=4, persistent_workers=True,
                                       worker_init_fn=_worker_gc_disable, collate_fn=collate)
  ```
- 安全性：worker 内主要分配 numpy 数组，由引用计数回收（非循环引用），`gc.disable()` 不会造成内存泄漏；
  解码在独立子进程中，不受影响。

### F3 · 固化「运行中禁止热编辑被 spawn 的解码模块」【必须】
- 文件：[data/datasets/video_decode.py](file:///home/jovyan/pfs/users/zhanqian.wu/code/model_exp/data/datasets/video_decode.py)
- 改动：`decode_isolated` 启动子进程前，把当前模块源码的 hash 写进 request；
  子进程 `child_main` 启动后回传自身源码 hash；不一致即抛 `MediaExecutorBugError` 并明确打印
  「解码模块在作业运行期间被修改」，**快速失败而不是训练到一半崩溃**。
  （沿用现有 `_child_error_category` → `executor_bug` → 立即 raise 的既有语义。）
- 备选（更轻量）：仅在 `video_decode.py` 顶部加 `# EDIT-DISABLE-AFTER-LAUNCH` 注释约定 + 启动时打印源码 hash 到日志，人工核对。
  **决策：采用 hash 校验**，因为它是唯一能自动防住这一类事故的手段。

### F4 · RoPE 修复的训练侧验证与续训策略 【必须，先验证再长跑】
- 先做一次短验证（不长跑），确认修复后的 RoPE 没有让模型输出崩坏：
  - 用 `stage_1p1/checkpoints/weights/step-00180000.safetensors` + 新 RoPE 在 512 桶做一次推理，
    与 `validation/step-00184000/rank0.png` 对比（修复前 GEN 语义已明显对齐中世纪城堡）。
  - 期望：结构仍连贯；语义可能短期退化（位置编码坐标系切换），**loss 短暂跳升属正常**。
- 决策：**不从头重训**。理由：网络参数全部可加载（139/139），错误 RoPE 提供的是「畸变但存在」的位置信息，
  模型能较快重新适应；从头重训代价远超收益。续训 2~5k 步后看 val 是否回到 184k 水平。
- 若 5k 步内语义不恢复 → 再考虑退回「修复 RoPE + 从 stage_1 的 192p 权重重新开始 512p 阶段」。

### F5 · 量化 `resize_long_edge` 补黑边比例 【建议，1% 采样】
- 抽样 `stage_1p1` 训练桶内 1% 样本，统计「补黑像素占比 > 1%」的样本比例。
- 若比例显著（>10%），在 `resize_long_edge` 与 `cover` 之间按源图比例自适应选择，或给桶增加更细的比例档。

### F6 · 文档同步
- [docs/giga_exp_architecture.md](file:///home/jovyan/pfs/users/zhanqian.wu/code/model_exp/docs/giga_exp_architecture.md)
  补充：RoPE `i ↔ i+64` 配对约束、`resize_long_edge` 补黑语义、跨 rank skip 的 all_reduce 同步协议。

---

## 4. 验证步骤（重启后按序执行）

1. `python -c "import data.datasets.video_decode as v; print(v.__file__)"` + 跑 F3 的 hash 校验自测：
   故意在子进程运行期间改文件 → 断言抛出 `executor_bug` 而非 TypeError。
2. 单机 8 卡跑 `stage_1_1.yaml` **20~50 步冒烟**：
   - 日志出现 `🔁 已恢复完整 DeepSpeed checkpoint step-180000`（F1 生效）
   - 日志出现 `📦 dataset ready: records=34,28x,xxx`，且**无** `resize_frames() got an unexpected keyword`
   - 观察 `ps` 中 worker RSS 不出现数量级跃升（F2 生效）
3. 集群 48 rank 跑 500~1000 步：
   - 日志目录为 `logs/<run_id>/node0..node5/`，各自有独立 `console.log`（F5/P5 生效）
   - 无 `TCPStore Broken pipe`；无 `🚫 本 step 的全部 micro batch 均因坏数据跳过`
4. 跑到 step=185000 触发 validation，比对 `validation/step-00185000/rank0.png`：
   - 语义对齐（对照 184000 的「中世纪城堡」是否依然正确）→ 判定 RoPE 重新适应是否成功

---

## 5. 不做的事（明确排除，避免过度工程）

- 不重写解码子进程架构、不引入共享内存索引、不把索引改成 mmap/Arrow 格式（内存目前够用）。
- 不改 `stage_1`（192p）正在运行的作业——它的代码路径与 stage_1p1 共享，若现在改 `video_decode.py` / `giga_transformer.py`
  会**同样影响正在跑的 192p 作业的解码子进程**（F3 的 hash 校验会让它立刻失败）。
  → **重要排期约束：任何对 `video_decode.py` / `giga_transformer.py` 的改动，
  必须等 `stage_1` 那一轮跑完，或接受 stage_1 被 F3 打断。**
- 不新增配置项去控制 `resize_long_edge` 的开关（YAML 里已有 `preprocess_style`，够用）。

---

## 6. 需要用户决策的两点

1. **F3 的落地方式**：hash 校验（自动防呆，但会让运行中的 `stage_1` 192p 作业在下一次解码时失败）
   vs 纯注释约定 + 启动日志打印 hash（不打断 stage_1，但靠人守规矩）。
2. **F4 的续训起点**：`stage_1p1` 自己的 `step-180000`（推荐，保留 512p 进展）
   vs `stage_1` 的 192p 权重从零重开 512p 阶段（更干净，但浪费已有进展）。
