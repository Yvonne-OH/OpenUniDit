# mod-adaln: text / image 双分支 AdaLN 调制

MiniMax H3 风格的模态独立调制：image token 与 text token 各用一套独立的 AdaLN
投影参数，替代原先全序列共享的单一 `mod`。**对旧 checkpoint 无损热启动**。

## 1. 设计

原实现（shared）：每层一个 `mod = nn.Linear(dim, dim*6)`，从 timestep 条件 `cond`
生成 6 组调制参数，对 image+text 拼接序列统一广播。

新实现（dual-branch，`mod_adaln: true`）：

```
mod_img(cond) → (shift_a, scale_a, gate_a, shift_f, scale_f, gate_f)  ← image 前缀
mod_txt(cond) → (shift_a, scale_a, gate_a, shift_f, scale_f, gate_f)  ← text  后缀
```

- 前向：序列按 `n_img` 切成两段，各用各分支的调制参数；attention/FFN 主体不变。
- 初始化继承：加载旧 ckpt 时，把旧 `blocks.{i}.mod.{weight,bias}` **拷贝**一份给
  `mod_img` 和 `mod_txt`。step0 时两分支输出完全相同 → forward 与 shared AdaLN
  **逐位相等**（已验证 `torch.equal` 级别），loss/生成不变。
- **拷贝复用 ≠ 权重共享**：两分支是独立 Parameter（两块内存），训练后各自接收
  梯度、慢慢分化成两套独立调制函数，没有参数共享、没有加性残差基底绑定。

数学上：

```
W_img = W_txt = W_old,  b_img = b_txt = b_old        (初始化)
step0:  mod_img(t) = mod_txt(t) = mod_old(t)         (行为与原版完全一致)
step k: W_img, W_txt 各自被独立梯度更新               (逐渐分化)
```

## 2. 配置开关

config 的 `model` 段：

```yaml
model:
  mod_adaln: true    # false 或缺省 = 原 shared 行为，完全不变
  ...
```

当前 `configs/` 下全部 10 个 yaml 均已开启 `mod_adaln: true`。

## 3. checkpoint 兼容与迁移警告

迁移函数 `models/giga_transformer.apply_mod_adaln_migration(current, state)` 在所有
权重加载路径（纯权重恢复 / SFT init_from / mHC strict 转换加载 / convert 工具）中
自动执行，三种情况：

| 情况 | 行为 | 警告 |
|---|---|---|
| yaml 开 dual，ckpt 只有旧 `mod.*` | 旧权重拷贝到两分支，无损等价热启动 | ⚠️ 黄色提示（层数列表） |
| yaml 开 dual，ckpt 连 `mod.*` 也没有 | 该层 `mod_img/mod_txt` 重新随机初始化 | ⚠️ 红色警告 |
| yaml 未开 dual，ckpt 是双分支权重 | `mod.*` 无法恢复、保持随机初始化 | ⚠️ 红色警告（提示改 yaml） |

注意 DeepSpeed 完整 state 恢复（含 optimizer）走 strict 加载，旧格式 state ckpt 会
直接失败并按既有逻辑**自动降级为纯权重恢复**（optimizer 状态重置，step 保留），
随后走上面的迁移路径。

`semantic_metadata` 新增 `mod_adaln` 字段：新旧 ckpt 混用时，metadata 对比会在启动时
额外报出 `mod_adaln: old=None, current=True` 一类的不匹配提示。

## 4. 改动文件

| 文件 | 内容 |
|---|---|
| `models/giga_transformer.py` | `GigaBlock` 双分支路径 + `apply_mod_adaln_migration` + `GigaExpTransformer(mod_adaln=)` + metadata |
| `models/giga_transformer_mhc.py` | mHC 变体同步支持（子层核心按 `n_img` 切分调制） |
| `scripts/train/train_giga_exp.py` | 加载迁移 + 警告播报 + banner |
| `scripts/train/train_giga_exp_kd.py` | 同上（student；teacher 不变） |
| `scripts/train/train_giga_exp_mhc.py` | strict 加载路径迁移 + banner |
| `tools/convert_checkpoint_to_mhc.py` | 转换工具兼容双分支 |
| `configs/*.yaml` ×10 | `model.mod_adaln: true` |

`scripts/train/*.sh` 无需改动：配置经 `GigaExpTransformer(**cfg["model"])` 自动透传。

## 5. 验证结果

- shared ckpt → dual 模型（拷贝迁移）：forward 输出与原 shared 模型 `torch.equal`
  级别完全相等（主模型与 mHC 变体均验证，maxdiff = 0.0）。
- 梯度独立：`mod_img.weight.grad` ≠ `mod_txt.weight.grad`，text 分支梯度非零。
- missing / orphan / 新格式 ckpt 三种迁移场景行为正确。
- 既有测试全部通过：`tests/test_function_preservation.py`、
  `tests/test_mhc_full_model.py`、`tests/test_checkpoint_conversion.py`。

## 6. 资源开销与推理兼容

- 参数量：每层由 1 个 mod 投影变为 2 个（`mod_img` + `mod_txt`），净增
  `6·dim² + 6·dim` ≈ 14.2M/层（dim=1536）；28 层模型合计约 +396M。
  这是有意为之的容量扩展；如需省参可改回 `mod_adaln: false`（迁移函数同样支持反向提示）。
- 推理：`pipeline/giga_pipeline.py`、`scripts/infer/*` 经 `**config["model"]` 透传，
  推理旧 ckpt 时请保持 `mod_adaln: false`（或开启后依赖迁移警告自行确认）。
