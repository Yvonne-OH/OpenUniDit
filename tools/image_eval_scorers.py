# -*- coding: utf-8 -*-
"""
无 GT（reference-free）图像评价指标集合。

指标清单（名称 / 是否需要 prompt / 量程 / 说明）：
    clip           ✅   ~0-35    CLIPScore = 2.5 × cos（CLIP ViT-L/14），图文对齐
    siglip         ✅   ~0-1     SigLIP so400m 图文余弦相似度
    imagereward    ✅   ~-2-3    人类偏好代理（ImageReward-v1.0）
    pickscore      ✅   ~10-30   人类偏好（PickScore_v1，logits_per_text）
    musiq          ❌   0-100    无参考图像质量（MUSIQ，koniq 版）
    liqe           ❌   1-5      质量 + 场景/类别（LIQE）
    laion_aes      ❌   0-10     美学（LAION Aesthetic Predictor V2）

权重获取：
    - CLIP(ViT-L/14)        本地 HF cache（openai/clip-vit-large-patch14）
    - SigLIP                google/siglip-so400m-patch14-384
    - PickScore             yuvalkirstain/PickScore_v1
    - ImageReward           zai-org/ImageReward → ~/.cache/ImageReward/
    - MUSIQ/LIQE/LAION-Aes  pyiqa 0.1.16（权重自动下载：HF chaofengc/IQA-PyTorch-Weights
                              或 GitHub release，二者均可达）
"""
import os

# 国内网络：hf-mirror 优先；必须在 import transformers / huggingface_hub 之前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from PIL import Image

SIGLIP_REPO = "google/siglip-so400m-patch14-384"
PICKSCORE_REPO = "yuvalkirstain/PickScore_v1"
IMAGEREWARD_REPO = "zai-org/ImageReward"        # 官方权重仓（迁移后的位置）
IMAGEREWARD_SNAPSHOT = "ImageReward-v1.0"        # pip 包期望的权重文件名
CLIP_MODEL = "openai/clip-vit-large-patch14"     # 本地 HF cache 已有


def _open_image(path: str) -> Image.Image:
    img = Image.open(path)
    img.load()
    return img.convert("RGB")


def _to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL → [1,C,H,W] float32 [0,1]（pyiqa 输入格式）"""
    import numpy as np
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


class BaseScorer:
    """评分器基类：__call__(paths, prompts) → 与输入等长的分数列表（失败项为 None）。"""

    name = "base"
    needs_prompt = False
    range_hint = ""

    def __init__(self, device: str = "cuda:0", batch_size: int = 8):
        self.device = device
        self.batch_size = batch_size

    def __call__(self, paths, prompts):
        raise NotImplementedError


class _ClipLikeScorer(BaseScorer):
    """CLIP / SigLIP / PickScore 共用的批量图文相似度骨架。"""

    def _load(self):
        raise NotImplementedError

    def _encode(self, imgs, prompts):
        raise NotImplementedError

    def __call__(self, paths, prompts):
        self._load()
        out = []
        for i in range(0, len(paths), self.batch_size):
            bp = prompts[i:i + self.batch_size]
            imgs = [_open_image(p) for p in paths[i:i + self.batch_size]]
            try:
                out.extend(self._encode(imgs, bp))
            except Exception as e:  # 单批失败不拖垮整体
                print(f"\033[33m⚠️ [{self.name}] 批内推理失败: {e}\033[0m")
                out.extend([None] * len(imgs))
        return out


class CLIPScorer(_ClipLikeScorer):
    """CLIPScore（Hessel et al. 2021）：2.5 × cosine(image_emb, text_emb)，ViT-L/14。"""
    name = "clip"
    needs_prompt = True
    range_hint = "~0-35（越高越对齐）"

    def _load(self):
        if hasattr(self, "model"):
            return
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(CLIP_MODEL).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    @torch.no_grad()
    def _encode(self, imgs, prompts):
        inputs = self.processor(text=prompts, images=imgs, return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
        outputs = self.model(**inputs)
        cos = (outputs.image_embeds.float() @ outputs.text_embeds.float().T) \
            .diagonal().cpu()
        return [round(2.5 * c.item(), 4) for c in cos]  # 标准 CLIPScore 权重 2.5


class SigLIPScorer(_ClipLikeScorer):
    """SigLIP so400m 图文余弦相似度（比 CLIP 更强的对齐度量）。"""
    name = "siglip"
    needs_prompt = True
    range_hint = "~0-1（越高越对齐）"

    def _load(self):
        if hasattr(self, "model"):
            return
        from transformers import SiglipModel, AutoProcessor
        self.model = SiglipModel.from_pretrained(SIGLIP_REPO).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(SIGLIP_REPO)

    @torch.no_grad()
    def _encode(self, imgs, prompts):
        inputs = self.processor(text=prompts, images=imgs, return_tensors="pt",
                                padding="max_length", truncation=True).to(self.device)
        outputs = self.model(**inputs)
        sims = torch.nn.functional.cosine_similarity(
            outputs.image_embeds.float(), outputs.text_embeds.float()).cpu()
        return [round(s.item(), 4) for s in sims]


class PickScoreScorer(_ClipLikeScorer):
    """PickScore（Pick-a-Pic 人类偏好打分），量程典型 10–30。"""
    name = "pickscore"
    needs_prompt = True
    range_hint = "~10-30（越高越偏好）"

    def _load(self):
        if hasattr(self, "model"):
            return
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(PICKSCORE_REPO).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(PICKSCORE_REPO)

    @torch.no_grad()
    def _encode(self, imgs, prompts):
        inputs = self.processor(text=prompts, images=imgs, return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits_per_image.diagonal().float().cpu()
        return [round(v.item(), 4) for v in logits]


class ImageRewardScorer(BaseScorer):
    """ImageReward-v1.0（人类偏好代理，可为负分）。

    注意：ImageReward 内置 BLIP 代码基于 transformers 4.15，与当前 transformers 5.x
    存在符号搬移/删除的不兼容。这里在本进程内做运行时兼容注入（_patch_transformers_v5），
    不修改 site-packages 中的任何文件，仅影响本评分进程。
    """
    name = "imagereward"
    needs_prompt = True
    range_hint = "~-2-3（越高越偏好）"

    @staticmethod
    def _patch_transformers_v5():
        import sys
        import types
        import transformers
        mu = transformers.modeling_utils
        try:
            import transformers.pytorch_utils as tpu
            for n in ("apply_chunking_to_forward", "prune_linear_layer"):
                if not hasattr(mu, n):
                    setattr(mu, n, getattr(tpu, n))
        except ImportError:
            pass
        if not hasattr(mu, "find_pruneable_heads_and_indices"):
            # v5 已删除（head pruning 机制移除）；按 4.x 官方实现补一个纯函数，
            # 推理路径永远不会真正调用它，仅为满足 med.py 的模块级 import
            import numpy as np

            def _fpih(heads, n_heads, head_size, already_pruned_heads):
                mask = np.ones(n_heads, dtype=bool)
                heads = list(set(heads) - already_pruned_heads)
                for head in heads:
                    head -= sum(mask[:head])
                    mask[head] = False
                index = torch.arange(len(mask))[torch.from_numpy(mask.astype(np.uint8))].long()
                return heads, index

            mu.find_pruneable_heads_and_indices = _fpih
        if not hasattr(transformers, "file_utils"):
            fu = types.ModuleType("transformers.file_utils")
            fu.ModelOutput = transformers.modeling_outputs.ModelOutput
            sys.modules["transformers.file_utils"] = fu
            transformers.file_utils = fu
        # BLIP init_tokenizer 依赖 v4 的 additional_special_tokens_ids 属性；
        # transformers 5 重构了 tokenizer（SpecialTokensMixin 移除），在进程内重写该函数。
        # score() 路径只用 tokenizer 编码 prompt，enc_token_id 不参与推理。
        try:
            import ImageReward.models.BLIP.blip as _blip_mod
            from transformers import BertTokenizer as _BertTok

            def _init_tokenizer_v5():
                tok = _BertTok.from_pretrained("bert-base-uncased")
                tok.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
                added = tok.get_added_vocab() if hasattr(tok, "get_added_vocab") else {}
                enc_id = added.get("[ENC]", tok.convert_tokens_to_ids("[ENC]"))
                tok.enc_token_id = enc_id
                return tok

            if not hasattr(_blip_mod.init_tokenizer, "_v5_patched"):
                _blip_mod.init_tokenizer = _init_tokenizer_v5
                _blip_mod.init_tokenizer._v5_patched = True
            # blip_pretrain 通过 `from .blip import init_tokenizer` 拷贝了引用，
            # 必须同时替换其模块命名空间里的绑定
            import ImageReward.models.BLIP.blip_pretrain as _bp_mod
            _bp_mod.init_tokenizer = _blip_mod.init_tokenizer
        except Exception:
            pass
        # med.py 的 BertModel 走 v4 风格 self.init_weights()，v5 的 tie_weights 需要
        # 若干初始化期才挂上的元数据属性，这里兜底补空值（med BertModel 无自动 tie 需求）
        try:
            _pm = transformers.modeling_utils.PreTrainedModel
            if not getattr(_pm.init_weights, "_v5_compat_patched", False):
                _orig_iw = _pm.init_weights

                def _iw_compat(self, *a, **kw):
                    for attr in ("all_tied_weights_keys", "_tp_plan",
                                 "_ep_plan", "_pp_plan"):
                        if not hasattr(self, attr):
                            setattr(self, attr, {})
                    return _orig_iw(self, *a, **kw)

                _pm.init_weights = _iw_compat
                _iw_compat._v5_compat_patched = True
        except Exception:
            pass
        # v5 移除了 PreTrainedModel.get_head_mask（head pruning 一并清理），
        # med.py 的 forward 在调用，按 v4 官方实现补回
        try:
            _pm2 = transformers.modeling_utils.PreTrainedModel
            if "get_head_mask" not in _pm2.__dict__:
                def _get_head_mask(self, head_mask, num_hidden_layers,
                                   is_attention_chunked=False):
                    if head_mask is not None:
                        if head_mask.dim() == 1:
                            head_mask = (head_mask.unsqueeze(0).unsqueeze(0)
                                         .unsqueeze(-1).unsqueeze(-1))
                            head_mask = head_mask.expand(
                                num_hidden_layers, -1, -1, -1, -1)
                        elif head_mask.dim() == 2:
                            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                        assert head_mask.dim() == 5
                        head_mask = head_mask.to(dtype=self.dtype)
                        if is_attention_chunked:
                            head_mask = head_mask.unsqueeze(-4)
                    else:
                        head_mask = [None] * num_hidden_layers
                    return head_mask

                _pm2.get_head_mask = _get_head_mask
        except Exception:
            pass

    @staticmethod
    def _ensure_weights() -> dict:
        """把 ImageReward.pt / med_config.json 预置到 pip 包的缓存目录（绕过其内置下载器）。"""
        import shutil
        from huggingface_hub import hf_hub_download
        root = os.path.join(os.path.expanduser("~"), ".cache", "ImageReward")
        os.makedirs(root, exist_ok=True)
        weight = os.path.join(root, IMAGEREWARD_SNAPSHOT + ".pt")
        med_cfg = os.path.join(root, "med_config.json")
        if not os.path.exists(weight):
            src = hf_hub_download(IMAGEREWARD_REPO, "ImageReward.pt")
            shutil.copy(src, weight)
        if not os.path.exists(med_cfg):
            src = hf_hub_download(IMAGEREWARD_REPO, "med_config.json")
            shutil.copy(src, med_cfg)
        # RM 内部还会加载 bert-base-uncased 分词器，顺手预热
        from transformers import BertTokenizer
        BertTokenizer.from_pretrained("bert-base-uncased")
        return {"weight": weight, "med_config": med_cfg}

    def _load(self):
        if hasattr(self, "model"):
            return
        self._patch_transformers_v5()
        import ImageReward as RM
        cache = self._ensure_weights()
        self.model = RM.load(IMAGEREWARD_SNAPSHOT, med_config=cache["med_config"],
                             device=self.device,
                             download_root=os.path.dirname(cache["weight"]))

    @torch.no_grad()
    def __call__(self, paths, prompts):
        self._load()
        out = []
        for p, pr in zip(paths, prompts):
            try:
                out.append(round(float(self.model.score(pr, p)), 4))
            except Exception as e:
                print(f"\033[33m⚠️ [imagereward] {os.path.basename(p)} 失败: {e}\033[0m")
                out.append(None)
        return out


class _PyIqaScorer(BaseScorer):
    """pyiqa 系（MUSIQ / LIQE / LAION-Aes）公共骨架。"""

    min_side = 0  # 输入短边下限（0 = 不限制）；不足时先 LANCZOS 上采样

    def __init__(self, metric_name: str, **kw):
        super().__init__(**kw)
        self.metric_name = metric_name

    @torch.no_grad()
    def __call__(self, paths, prompts):
        if not hasattr(self, "metric"):
            import pyiqa
            self.metric = pyiqa.create_metric(self.metric_name,
                                              device=torch.device(self.device))
        out = []
        for p in paths:
            try:
                img = _open_image(p)
                if self.min_side and min(img.size) < self.min_side:
                    k = self.min_side / min(img.size)
                    img = img.resize((round(img.width * k),
                                      round(img.height * k)), Image.LANCZOS)
                t = _to_tensor(img).to(self.device)
                v = self.metric(t).view(-1).float().cpu()
                out.append(round(v.item(), 4))
            except Exception as e:
                print(f"\033[33m⚠️ [{self.name}] {os.path.basename(p)} 失败: {e}\033[0m")
                out.append(None)
        return out


class MusiqScorer(_PyIqaScorer):
    name = "musiq"
    needs_prompt = False
    range_hint = "0-100（无参考质量）"

    def __init__(self, **kw):
        super().__init__("musiq", **kw)


class LiqeScorer(_PyIqaScorer):
    name = "liqe"
    needs_prompt = False
    range_hint = "1-5（质量+场景）"
    min_side = 224  # LIQE 内部 CLIP ViT-B/32 要求短边 ≥224，不足时先上采样

    def __init__(self, **kw):
        super().__init__("liqe", **kw)


class LaionAesScorer(_PyIqaScorer):
    name = "laion_aes"
    needs_prompt = False
    range_hint = "0-10（美学）"

    def __init__(self, **kw):
        super().__init__("laion_aes", **kw)


# 指标注册表：名称 → 类（懒加载，首次调用才建模型）
SCORER_REGISTRY = {
    "clip": CLIPScorer,
    "siglip": SigLIPScorer,
    "imagereward": ImageRewardScorer,
    "pickscore": PickScoreScorer,
    "musiq": MusiqScorer,
    "liqe": LiqeScorer,
    "laion_aes": LaionAesScorer,
}

METRIC_INFO = {
    "clip":        ("CLIPScore",       "image-text alignment",     "⭐⭐⭐⭐"),
    "siglip":      ("SigLIP 相似度",   "image-text alignment",     "⭐⭐⭐⭐"),
    "imagereward": ("ImageReward",     "人类偏好代理",              "⭐⭐⭐⭐"),
    "pickscore":   ("PickScore",       "人类 preference",           "⭐⭐⭐⭐"),
    "musiq":       ("MUSIQ",           "无参考图像质量",            "⭐⭐⭐⭐"),
    "liqe":        ("LIQE",            "quality + scene/category", "⭐⭐⭐"),
    "laion_aes":   ("LAION Aesthetic", "美学",                      "⭐⭐⭐"),
    "vlm":         ("VLM API 评价",    "美学/技术/对齐综合",         "⭐⭐⭐⭐"),
}

ALL_MODEL_METRICS = list(SCORER_REGISTRY.keys())
