# text_trans: Text Refiner（文本前置精炼层）

MOSS-Image 风格的 Text Refiner：文本 token 在进入主 DiT 之前，先经过若干层
**恒等初始化**的残差自注意力 + FFN 精炼。**对旧 checkpoint 无损热启动**，
step 0 推理行为与原版逐位一致，之后平滑增量学习。

## 1. 设计

基线链路：

```
Text Encoder → text_in 投影 → concat → Main DiT (shared/dual AdaLN)
```

改造后链路（`text_trans` 开启时）：

```
Text Encoder → text_in 投影 → Text Refiner (残差自Attn+FFN ×N) → concat → Main DiT (结构完全不变)
```

模块特性（`models/giga_transformer.py: TextRefinerBlock / TextRefiner`）：

- 只处理文本 token，不与图像 token 交互；无 timestep AdaLN（文本无条件时序调制）；
- 每层结构：`LayerNorm → MultiheadAttention → 残差` + `LayerNorm → FFN(dim→4·dim) → 残差`；
- **恒等初始化**：残差支路输出层（`attn.out_proj` 与 FFN 末层）的 weight 与 bias 全部
  置零 → 初始 forward 是恒等映射；
- 文本 padding mask 透传（MHA `key_padding_mask`），padding 位置的输出会被主 DiT 的
  attention mask / FA3 varlen 丢弃，对数值无影响；
- 主干 `blocks / patch_in / out / rope / mod_adaln` 等全部不动，主干结构完全不变。

数学上：

```
W_out_proj = 0, b_out_proj = 0, W_ffn_last = 0, b_ffn_last = 0   (初始化)
step0:  refiner(x) = x + 0·attn(...) + 0·ffn(...) = x            (与原版逐位一致)
step k: 残差支路权重接收梯度, 从零平滑长出文本精炼能力          (平滑增量学习)
```

> 注意：恒等初始化 ≠ 权重冻结。残差支路虽全零，但 `in_proj` / FFN 首层是随机初始化的，
> 一开始训练 refiner 即立刻获得非零梯度并开始分化（zero-init 的输出层对梯度无碍）。

## 2. 配置开关

config 的 `model` 段，三种写法：

```yaml
model:
  text_trans: true                # 最简: 2 层, heads 同主 DiT
  text_trans: false                # 或缺省 = 原版直通, 完全不变
  text_trans:
    enabled: true
    layers: 2                      # MOSS 配置 2 层; 消融可先改 1
    heads: 14                      # 默认同主 DiT heads
```

当前 `configs/` 下**所有 10 个 yaml 均已默认开启**
`text_trans: {enabled: true, layers: 2, heads: <同主 DiT>}`（含 mhc_2stream / mhc_from0）；
关闭设 `enabled: false` 或删除该段即回原版直通。

`GigaExpTransformerMHC`（`models/giga_transformer_mhc.py`）同样支持 `text_trans`
（复用同一个 `TextRefiner`，接在 `text_in` 之后、`stack_stream` 之前，mHC 残差结构不变），
mHC 训练脚本复用主脚本的 `load_pure_weights` → 自动获得相同的恒等初始化 + 警告逻辑；
`tools/convert_checkpoint_to_mhc.py` 对旧 ckpt（无 `text_refiner.*`）转换时在输出中补零初始化
refiner 并警告，对已带 `text_refiner.*` 的 ckpt 则原样保留。

## 3. checkpoint 兼容与警告

所有纯权重加载路径（`load_pure_weights` / `load_init_weights`，即 `resume_mode` 指定恢复、
SFT `init_from`、完整 state 恢复失败后的自动降级）都走 `load_compatible_state_dict`：

| 情况 | 行为 | 提示 |
|---|---|---|
| yaml 开 text_trans，ckpt 无 `text_refiner.*` | refiner 保持恒等映射零初始化，其余权重正常加载 | ⚠️ 黄色警告（tensor 数 / 参数量） |
| yaml 开 text_trans，ckpt 有 `text_refiner.*` | 正常加载，无警告 | 无 |
| yaml 未开，ckpt 带 `text_refiner.*` | refiner 权重被忽略（名称匹配不上），模型为原版结构 | semantic metadata mismatch 警告 |

警告示例（`train_giga_exp.py / train_giga_exp_kd.py: _report_text_trans_reinit`）：

```
⚠️ text_trans: checkpoint 缺少 text_refiner.* 权重但 yaml model.text_trans 已开启,
  24 个 tensor / 77.12M 参数保持恒等映射零初始化 (step0 forward 与原版一致, 平滑增量学习)
```

`semantic_metadata` 已加入 `"text_trans": {"layers": 2, "heads": 14}` 字段，新旧 ckpt 混用时
启动期会自动报出元数据差异，便于追溯 ablation 配置。

注意 DeepSpeed 完整 state 恢复（含 optimizer）要求 ckpt 内含 `text_refiner.*` 分片；
旧格式 state ckpt 会 strict 失败并按既有逻辑**自动降级为纯权重恢复**（optimizer 重置），
随后走上面的恒等初始化路径。

## 4. 参数量（dim=1792, heads=14）

| 组件 | 每层参数 | 2 层合计 |
|---|---|---|
| attn in_proj (3·d² + 3d) | 9,639,168 | 19,278,336 |
| attn out_proj (d² + d) | 3,213,056 | 6,426,112 |
| FFN (d·4d + 4d + 4d·d + d) | 25,699,072 | 51,398,144 |
| 2 × LayerNorm | 7,168 | 14,336 |
| **合计** | **38,558,464** | **77,116,928 (77.12M)** |

- 主 DiT 基线 ≈1110.37M → 加 2 层 refiner 后 1187.48M（**+77.12M, +6.9%）**；
- dim=1536 / heads=12 配置（stage_1 系列 / mhc 系列）：每层 28,331,520，2 层
  **56,663,040 (56.66M)**；
- 若消融用 1 层：对应减半。

## 5. step0 恒等验证

`GigaExpTransformer(text_trans=2层)` 用旧 ckpt 权重热启动（refiner 保持零初始化）后，
与 `text_trans=False` 的原版模型在相同输入（含右 padding mask）下：

```
step0 max delta (fp32, 带 padding mask): 0.000e+00
```

即逐位相等（验证脚本逻辑：同一权重、同一输入、开/关 refiner 对比输出）。
FLOPs 增量仅 `O(L_txt²·d)` 的文本自注意力 + FFN（L_txt ≤ 512，远小于图像 token 数），
推理/训练吞吐影响可忽略。

## 6. 训练脚本改动

`scripts/train/train_giga_exp.py` 与 `scripts/train/train_giga_exp_kd.py`（两处同构）：

1. `load_compatible_state_dict` 内新增 `_report_text_trans_reinit(model, state)`：
   检测「模型有 `text_refiner.*` 但 ckpt 没有」→ 保持恒等初始化 + rank0 黄色警告；
2. `print_banner` 新增 `✍️ Text Refiner(text_trans)` 一行，启动即可见开关状态与层数。

`scripts/train/train_giga_exp_mhc.py`：banner 同款状态行；权重加载复用主脚本的
`load_pure_weights`（自动带恒等初始化 + 警告）。mHC 转换权重路径保持 strict
（转换输出的 ckpt 已含零初始化 `text_refiner.*`，键集合一致可直接 strict）。
