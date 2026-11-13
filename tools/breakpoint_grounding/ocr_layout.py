"""breakpoint-grounding-55m stage 3: OCR (PP-OCRv5) + Qwen3-VL 结构化布局标注, 打包 parquet & mdb。

layout JSON (MOSS-Image 风格):
  summary: {meta, canvas:"single", caption=全图 t2i 描述 (Qwen3-VL, 复用 stage2 已打的)}
  items:
    text   小框 (长边 < 图长边×vlm_min_frac): text=OCR 原文, quad=OCR 旋转四点原样,
           caption=默认描述 (九宫格位置 + OCR 文本);
           大框 (长边 >= 图长边×vlm_min_frac): 送一次 VLM 返回 JSON {text, bbox, style}:
             text  = 相似度>=0.8 用 OCR 原文, 否则 VLM 转写;
             quad  = VLM 框 (crop 内归一化坐标映射回原图) 与 OCR 框融合 (有重叠取并集) 的四角;
             caption = VLM 的文字外观描述;
           text item 只带 "quad" 旋转四边形, 不带水平 bbox;
    object label: 数据集自带 object_captions + normalized_boxes 直接复用, bbox=轴对齐框;
    bbox 坐标空间 abs_centralized (原点图像中心, 单位 px), 格式 yxyx [y_min,x_min,y_max,x_max];
    quad 同坐标空间, 格式 [[y,x]×4] 顺时针从左上角开始 (quad_format="yx4_cw")。
  item 级过滤: min(y_len,x_len) < bbox_min_side 丢弃; 同 label IoS > overlap_th 只留大框。

全量模式 (--num-gpus 8): 每卡一个 worker 消费分片队列, 只处理 stage1 质量过滤合格的行
  (阈值同 stage2, 可 CLI 调), 输出 GenPoster100K 格式:
    <root>/ocr_layout/parquet/<stem>.parquet: sample_id/source_index/caption/caption_zh/ocr/
      width/height/file_size/caption_ok/orig_id/layout(JSON str)
    <root>/ocr_layout/mdb/<stem>.mdb: key=sample_id uint64 LE, value=pickle dict
      {id, source, caption, caption_zh, ocr(list), w, h, img_bytes, caption_ok, orig_id, layout}
  sample_id = shard_index<<20 | row_index; 分片级断点续跑; --merge 合并出 meta.parquet。

用法 (先出 sample case 供人工确认):
  source /home/jovyan/pfs/users/zhanqian.wu/code/env/gwp.sh
  python ocr_layout.py --sample 8 --scan 128   # 扫前 128 行, 选文本最多的 8 张出样例
  python ocr_layout.py --num-gpus 8            # 全量 8 卡打包
  python ocr_layout.py --status / --merge
sample 输出: /home/jovyan/pfs/users/zhanqian.wu/exp/breakpoint_grounding/layout_sample/
"""
from __future__ import annotations

import argparse
import base64
import difflib
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch  # VLM.ask 的 @torch.inference_mode() 装饰器在类定义时求值

ROOT_DEFAULT = "/mnt/nfs/data/zhanqian.wu/unified_canvas_images/breakpoint-grounding-55m"
MODEL_PATH = "/mnt/nfs/models/huggingface/Qwen/Qwen3-VL-4B-Instruct"
OUT_DIR = Path("/home/jovyan/pfs/users/zhanqian.wu/exp/breakpoint_grounding/layout_sample")
OUT_SUBDIR = "ocr_layout"
# PP-OCRv5 本地模型 (预下载自 bcebos, 避免 paddlex 运行时下载卡住)
OCR_DET_DIR = "/home/jovyan/pfs/users/zhanqian.wu/exp/breakpoint_grounding/ppocrv5_models/PP-OCRv5_server_det_infer"
OCR_REC_DIR = "/home/jovyan/pfs/users/zhanqian.wu/exp/breakpoint_grounding/ppocrv5_models/PP-OCRv5_server_rec_infer"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from caption_qwen3vl import DEFAULT_THRESHOLDS as QUALITY_DEFAULTS, load_filter_rows, row_pass  # noqa: E402

T2I_PROMPT = (
    "Describe this image as a text-to-image prompt in one paragraph of about 80-100 words. "
    "Cover the subject and its attributes, composition and framing, art or photographic "
    "style, colors, lighting and atmosphere. Output only the description itself."
)
# 大 text 框: 一次调用同时拿 转写+定位+外观描述
LARGE_REGION_PROMPT = (
    "This is a cropped text region from a larger image. Respond ONLY with a JSON object with 3 keys: "
    '1. "text": transcribe ALL text visible in the crop, exactly as written; '
    '2. "bbox": tight bounding box of the complete text block within THIS crop, normalized to 0-1, '
    'as [y0, x0, y1, x1] measured from the crop\'s top-left corner; '
    '3. "style": under 25 words describing the text\'s visual style (font size/weight, color, '
    'casing, and what kind of text it is: title, subtitle, label, watermark, URL...). '
    'JSON only, no markdown.'
)

LAYOUT_BBOX_DEFAULTS = dict(bbox_min_side=16, overlap_th=0.5, vlm_min_frac=0.18)


# ─────────────────────────── OCR (PP-OCRv5) ───────────────────────────

def build_ocr(device: str = "gpu", **kwargs):
    from paddleocr import PaddleOCR
    return PaddleOCR(
        device=device,
        text_detection_model_name="PP-OCRv5_server_det",
        text_detection_model_dir=OCR_DET_DIR,
        text_recognition_model_name="PP-OCRv5_server_rec",
        text_recognition_model_dir=OCR_REC_DIR,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        cpu_threads=8,
        rec_batch_num=32,
        text_det_limit_side_len=960,  # 默认 limit_type=min 会放大小图; max 限长边, 快一倍
        text_det_limit_type="max",
        **kwargs,
    )


def ocr_image(ocr, img_bgr: np.ndarray) -> list[dict]:
    """→ [{text, quad(4x2), det_score, rec_score}]; det/rec 任一缺失记 nan。"""
    results = ocr.predict(img_bgr)
    lines: list[dict] = []
    for res in results:
        d = res if isinstance(res, dict) else getattr(res, "json", {}).get("res", {})
        texts = d.get("rec_texts") or []
        polys = d.get("rec_polys") or d.get("dt_polys") or []
        rec_scores = d.get("rec_scores") or []
        det_scores = d.get("dt_scores") or d.get("det_scores") or []
        for i, text in enumerate(texts):
            quad = np.asarray(polys[i], dtype=np.float32).reshape(-1, 2) if i < len(polys) else None
            if quad is None or len(quad) != 4:
                continue
            lines.append({
                "text": text,
                "quad": quad.tolist(),
                "det_score": float(det_scores[i]) if i < len(det_scores) else float("nan"),
                "rec_score": float(rec_scores[i]) if i < len(rec_scores) else float("nan"),
            })
    return lines


# ─────────────────────────── 坐标 / 过滤规则 ───────────────────────────

def quad_to_yxyx(quad, w: int, h: int) -> list[int]:
    """4 点旋转框 → 外接轴对齐框, abs_centralized yxyx [y_min, x_min, y_max, x_max]。"""
    q = np.asarray(quad, dtype=np.float32)
    x0, y0 = q[:, 0].min(), q[:, 1].min()
    x1, y1 = q[:, 0].max(), q[:, 1].max()
    return [int(round(y0 - h / 2)), int(round(x0 - w / 2)),
            int(round(y1 - h / 2)), int(round(x1 - w / 2))]


def quad_to_yx4(quad, w: int, h: int) -> list[list[float]]:
    """4 点旋转框 (quad 行为 [x,y] 绝对坐标) → abs_centralized [[y,x]×4] 顺时针从左上。"""
    q = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    s = q[:, 1] + q[:, 0]  # y+x 最小者为左上角
    q = np.roll(q, -int(np.argmin(s)), axis=0)
    return [[round(float(y) - h / 2, 1), round(float(x) - w / 2, 1)] for x, y in q]


def yxyx_to_yx4(b: list[int]) -> list[list[int]]:
    """轴对齐框四角 (顺时针从左上)。"""
    y0, x0, y1, x1 = b
    return [[y0, x0], [y0, x1], [y1, x1], [y1, x0]]


def norm_box_to_yxyx(b: list[float], w: int, h: int) -> list[int]:
    """数据集 normalized_boxes (xyxy, [0,1]) → abs_centralized yxyx。"""
    x0, y0, x1, y1 = b
    return [int(round(y0 * h - h / 2)), int(round(x0 * w - w / 2)),
            int(round(y1 * h - h / 2)), int(round(x1 * w - w / 2))]


def box_len(b) -> tuple[int, int]:
    return int(b[2]) - int(b[0]), int(b[3]) - int(b[1])  # y_len, x_len


def iou(a, b) -> float:
    iy = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ix = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iy * ix
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


def union_box(a, b) -> list[int]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def position_phrase(b, w: int, h: int) -> str:
    """框中心在画面九宫格中的位置短语。"""
    cy = (b[0] + b[2]) / 2 / h
    cx = (b[1] + b[3]) / 2 / w
    rows, cols = ["top", "center", "bottom"], ["left", "center", "right"]
    ri = 0 if cy < 1 / 3 else (1 if cy < 2 / 3 else 2)
    ci = 0 if cx < 1 / 3 else (1 if cx < 2 / 3 else 2)
    r, c = rows[ri], cols[ci]
    if r == "center" and c == "center":
        return "centered"
    if r == "center":
        return f"centered {c}"
    if c == "center":
        return f"at the {r}"
    return f"at the {r} {c}"


def overlap_dedup(items: list[dict], overlap_th: float) -> list[dict]:
    """同 label 内按面积降序贪心保留; IoS = 交面积/较小框面积 > overlap_th 时丢小框。"""
    kept: list[dict] = []
    for it in sorted(items, key=lambda x: -np.prod(box_len(x["bbox"]))):
        a = it["bbox"]
        dup = False
        for k in kept:
            if k["label"] != it["label"]:
                continue
            iy = max(0, min(a[2], k["bbox"][2]) - max(a[0], k["bbox"][0]))
            ix = max(0, min(a[3], k["bbox"][3]) - max(a[1], k["bbox"][1]))
            smaller = min(np.prod(box_len(a)), np.prod(box_len(k["bbox"])))
            if iy * ix / max(1, smaller) > overlap_th:
                dup = True
                break
        if not dup:
            kept.append(it)
    return kept


# ─────────────────────────── VLM (Qwen3-VL-4B) ───────────────────────────

class VLM:
    def __init__(self, max_pixels: int = 768 * 768):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
        self.processor.image_processor.max_pixels = max_pixels
        self.processor.image_processor.min_pixels = 256 * 256
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa").to("cuda").eval()

    @torch.inference_mode()
    def ask(self, images: list, prompts: list[str], max_new_tokens: int = 256) -> list[str]:
        texts = [self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}],
            tokenize=False, add_generation_prompt=True) for p in prompts]
        inputs = self.processor(text=texts, images=[[im] for im in images], padding=True,
                                return_tensors="pt").to("cuda")
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return [t.strip() for t in self.processor.batch_decode(
            out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)]


def crop_quad(img: "PIL.Image.Image", quad, pad: float = 0.15):
    """crop 带 pad, 返回 (PIL crop, 原图坐标 (cx0, cy0, cx1, cy1)); 退化 quad 返回 (None, None)。"""
    w, h = img.size
    q = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    qx, qy = np.sort(q[:, 0]), np.sort(q[:, 1])
    px, py = (qx[-1] - qx[0]) * pad, (qy[-1] - qy[0]) * pad
    box = (max(0, int(qx[0] - px)), max(0, int(qy[0] - py)), min(w, int(qx[-1] + px)), min(h, int(qy[-1] + py)))
    if box[2] <= box[0] or box[3] <= box[1]:
        return None, None
    return img.crop(box), box


def parse_vlm_json(s: str) -> dict | None:
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def text_sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


# ─────────────────────────── layout 组装 ───────────────────────────

def prepare_text_lines(lines: list[dict], w: int, h: int, th: dict) -> list[tuple]:
    """OCR 行 → [(ln, bbox_yxyx, is_large)], 过滤空转写与过小框。"""
    text_lines = []
    long_side = max(w, h)
    for ln in lines:
        if not ln["text"].strip():  # OCR 空转写直接丢
            continue
        bbox = quad_to_yxyx(ln["quad"], w, h)
        if min(box_len(bbox)) < th["bbox_min_side"]:
            continue
        text_lines.append((ln, bbox, max(box_len(bbox)) >= th["vlm_min_frac"] * long_side))
    return text_lines


def collect_large_crops(pil_img, text_lines: list[tuple], pad: float = 0.15) -> list[tuple]:
    """收集大框 crop → [(k, PIL crop, crop_box)], k 为 text_lines 下标 (仅 is_large 项)。"""
    crops = []
    for k, (ln, bbox, is_large) in enumerate(text_lines):
        if is_large:
            c, box = crop_quad(pil_img, ln["quad"], pad)
            if c is None:  # 退化 quad 跳过, 该大框走默认路径
                continue
            crops.append((k, c, box))
    return crops


def build_layout(row: dict, text_lines: list[tuple], w: int, h: int, th: dict, meta: list[str],
                 summary_caption: str, vlm_out: dict | None = None) -> dict:
    """组装 layout JSON。vlm_out: {k: (vlm_json, crop_box)} 只含大框 (无则大框走默认路径)。"""
    vlm_out = vlm_out or {}
    items: list[dict] = []
    for k, (ln, bbox, is_large) in enumerate(text_lines):
        pos = position_phrase(bbox, w, h)
        entry = vlm_out.get(k)
        if entry is None or not entry[0]:
            # 小框 / 无 VLM 结果: OCR 原文 / OCR 框 (含旋转四点) / 默认描述
            items.append({"label": "text", "caption": f"{pos}: text '{ln['text']}'.",
                          "text": ln["text"], "bbox": bbox,
                          "quad": quad_to_yx4(ln["quad"], w, h)})
            continue
        j, crop_box = entry
        vlm_text = str(j.get("text") or "").strip()
        style = str(j.get("style") or "").strip()
        # text: 相似度>=0.8 用 OCR 原文
        merged_text = ln["text"] if text_sim(vlm_text, ln["text"]) >= 0.8 else (vlm_text or ln["text"])
        # bbox: VLM 框与 OCR 框融合 (有重叠取并集), 否则保留 OCR 框
        if isinstance(j.get("bbox"), list) and len(j["bbox"]) == 4 \
                and all(isinstance(v, (int, float)) for v in j["bbox"]):
            cx0, cy0, cx1, cy1 = crop_box
            nx0, ny0, nx1, ny1 = [min(max(float(v), 0.0), 1.0) for v in j["bbox"]]
            vlm_bbox = [int(round(cy0 + ny0 * (cy1 - cy0)) - h / 2),
                        int(round(cx0 + nx0 * (cx1 - cx0)) - w / 2),
                        int(round(cy0 + ny1 * (cy1 - cy0)) - h / 2),
                        int(round(cx0 + nx1 * (cx1 - cx0)) - w / 2)]
            if box_len(vlm_bbox)[0] > 0 and box_len(vlm_bbox)[1] > 0 and iou(vlm_bbox, bbox) > 0.2:
                bbox = union_box(vlm_bbox, bbox)
        # caption: VLM 外观描述
        if style:
            style = style[0].lower() + style[1:] if style[0].isupper() else style
            items.append({"label": "text", "caption": f"{style.rstrip('.')}, {pos}.",
                          "text": merged_text, "bbox": bbox, "quad": yxyx_to_yx4(bbox)})
        else:
            items.append({"label": "text", "caption": f"{pos}: text '{merged_text}'.",
                          "text": merged_text, "bbox": bbox, "quad": yxyx_to_yx4(bbox)})

    for cap, b in zip(row["object_captions"], row["normalized_boxes"]):
        bbox = norm_box_to_yxyx(list(b), w, h)
        if min(box_len(bbox)) < th["bbox_min_side"]:
            continue
        items.append({"label": "object", "caption": cap, "bbox": bbox})

    items = overlap_dedup(items, th["overlap_th"])
    items.sort(key=lambda x: -np.prod(box_len(x["bbox"])))
    for it in items:
        if it["label"] == "text":
            it.pop("bbox")  # text 只留旋转 quad, 不出水平框
    return {
        "summary": {"meta": meta, "canvas": "single", "caption": summary_caption},
        "items": items,
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
        "quad_format": "yx4_cw",
    }


# ─────────────────────────── sample 驱动 ───────────────────────────

def draw_vis(img_bytes: bytes, items: list[dict], lines: list[dict]) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    def to_abs_xy(pt):  # [[y,x] centralized → [x,y] abs
        return [int(pt[1] + w / 2), int(pt[0] + h / 2)]

    for ln in lines:  # OCR 4 点旋转框: 红
        pts = np.asarray(ln["quad"], np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, (0, 0, 255), 2)
    for i, it in enumerate(items):
        color = (0, 255, 0) if it["label"] == "text" else (255, 128, 0)
        if it["label"] == "text":  # text: 旋转四边形
            pts = np.asarray([to_abs_xy(p) for p in it["quad"]], np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, color, 2)
        else:  # object: 轴对齐框
            y0, x0, y1, x1 = [v + (w // 2 if j % 2 else h // 2) for j, v in enumerate(it["bbox"])]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        if it["label"] == "text":  # label 标在 quad 左上角
            y0 = int(min(p[0] for p in it["quad"]) + h / 2)
            x0 = int(min(p[1] for p in it["quad"]) + w / 2)
        else:
            y0, x0 = it["bbox"][0] + h // 2, it["bbox"][1] + w // 2
        cv2.putText(img, f"{i}:{it['label']}", (x0, max(14, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def run_sample(args):
    import pyarrow.parquet as pq

    root = Path(args.root)
    shards = sorted((root / "data").glob("train-*.parquet"))
    shard = shards[args.shard_index]
    th = dict(LAYOUT_BBOX_DEFAULTS, bbox_min_side=args.bbox_min_side, overlap_th=args.overlap_th,
              vlm_min_frac=args.vlm_min_frac)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("🔧 加载 PP-OCRv5 (GPU) ...", flush=True)
    ocr = build_ocr()
    table = pq.read_table(shard).slice(args.row_offset, args.scan)
    rows = table.to_pylist()
    del table
    print(f"📖 {shard.name} 行 {args.row_offset}~{args.row_offset + len(rows)}, OCR 扫描 ...", flush=True)

    t0 = time.time()
    all_lines, valid_rows = [], []
    for i, row in enumerate(rows):
        img = cv2.imdecode(np.frombuffer(row["image"]["bytes"], np.uint8), cv2.IMREAD_COLOR)
        lines = ocr_image(ocr, img) if img is not None else []
        all_lines.append(lines)
        valid_rows.append((i, len(lines)))
        if (i + 1) % 16 == 0:
            print(f"  OCR {i+1}/{len(rows)} ({(i+1)/(time.time()-t0):.1f} img/s)", flush=True)
    n_with_text = sum(1 for _, n in valid_rows if n > 0)
    print(f"✅ OCR 完成: {len(rows)/(time.time()-t0):.2f} img/s, {n_with_text}/{len(valid_rows)} 张含文本", flush=True)

    with_text = sorted([x for x in valid_rows if x[1] > 0], key=lambda x: -x[1])
    no_text = [x for x in valid_rows if x[1] == 0]
    picked = [i for i, _ in with_text[: args.sample]]
    picked += [i for i, _ in no_text[: args.include_no_text]]

    print("🧠 加载 Qwen3-VL-4B (summary caption + 大框 转写/定位/描述) ...", flush=True)
    vlm = VLM()

    results = []
    for rank, i in enumerate(picked):
        row, lines = rows[i], all_lines[i]
        from PIL import Image
        pil = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        summary_cap = vlm.ask([pil], [T2I_PROMPT], max_new_tokens=200)[0]
        img = cv2.imdecode(np.frombuffer(row["image"]["bytes"], np.uint8), cv2.IMREAD_COLOR)
        h_img, w_img = img.shape[:2]
        text_lines = prepare_text_lines(lines, w_img, h_img, th)
        vlm_out = {}
        all_crops = collect_large_crops(pil, text_lines)
        for s in range(0, len(all_crops), 32):
            chunk = all_crops[s:s + 32]
            outs = vlm.ask([c for _, c, _ in chunk], [LARGE_REGION_PROMPT] * len(chunk))
            for (k, _, box), out in zip(chunk, outs):
                j = parse_vlm_json(out)
                if j and isinstance(j.get("bbox"), list) and len(j["bbox"]) == 4 \
                        and all(isinstance(v, (int, float)) for v in j["bbox"]):
                    vlm_out[k] = (j, box)
        layout = build_layout(row, text_lines, w_img, h_img, th, [args.meta], summary_cap, vlm_out)
        results.append(dict(row_index=args.row_offset + i, layout=layout,
                            ocr_lines=lines, img_bytes=row["image"]["bytes"]))
        n_text = sum(1 for it in layout["items"] if it["label"] == "text")
        n_obj = sum(1 for it in layout["items"] if it["label"] == "object")
        print(f"  [{rank}] row {args.row_offset+i}: {len(lines)} OCR 行 → layout {n_text} text + {n_obj} object", flush=True)

    html_rows = []
    for r in results:
        stem = f"row{r['row_index']:05d}"
        vis = draw_vis(r["img_bytes"], r["layout"]["items"], r["ocr_lines"])
        cv2.imwrite(str(OUT_DIR / f"{stem}_vis.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        (OUT_DIR / f"{stem}.json").write_text(
            json.dumps({"layout": r["layout"], "ocr_lines": r["ocr_lines"]}, ensure_ascii=False, indent=2))
        layout_json = json.dumps(r["layout"], ensure_ascii=False, indent=1)
        html_rows.append(
            f'<div style="display:flex;gap:16px;margin:16px 0;padding:12px;border:1px solid #ddd;border-radius:8px">'
            f'<img src="{stem}_vis.jpg" style="max-width:480px;border-radius:4px;align-self:flex-start">'
            f'<pre style="font-size:11px;white-space:pre-wrap;flex:1;margin:0">{layout_json}</pre></div>')
        print(f"💾 {stem}: {len(r['ocr_lines'])} OCR 行 / {len(r['layout']['items'])} items", flush=True)

    html = ('<!doctype html><meta charset="utf-8"><title>layout sample</title>'
            '<body style="font-family:sans-serif;max-width:1200px;margin:auto">'
            f'<h2>breakpoint-grounding-55m OCR + layout 样例</h2>'
            '<p style="color:#888">红=OCR 4点旋转框, 绿=layout text bbox, 蓝=layout object bbox</p>'
            + "".join(html_rows) + "</body>")
    (OUT_DIR / "index.html").write_text(html)
    print(f"\n🖼 预览页: {OUT_DIR / 'index.html'}", flush=True)


# ─────────────────────────── 全量 8 卡打包 ───────────────────────────

class Progress:
    def __init__(self, root: Path):
        self.path = root / OUT_SUBDIR / "_progress.json"
        self.lock_path = root / OUT_SUBDIR / "_progress.lock"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def done_shards(self) -> set[str]:
        if not self.path.exists():
            return set()
        return set(json.loads(self.path.read_text()).keys())

    def mark_done(self, stem: str, rows: int, seconds: float, worker: int):
        import fcntl
        with open(self.lock_path, "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(self.path.read_text()) if self.path.exists() else {}
            data[stem] = dict(rows=rows, seconds=round(seconds, 1), worker=worker, ts=time.time())
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
            os.replace(tmp, self.path)

    def snapshot(self) -> dict:
        return json.loads(self.path.read_text()) if self.path.exists() else {}


META_SCHEMA = [
    ("sample_id", "int64"), ("source_index", "int64"), ("caption", "string"), ("caption_zh", "string"),
    ("ocr", "string"), ("width", "int32"), ("height", "int32"), ("file_size", "int64"),
    ("caption_ok", "bool"), ("orig_id", "string"), ("layout", "string"),
]


def write_shard_outputs(out_dir: Path, stem: str, records: list[dict]):
    """records: {sample_id, source_index, caption, caption_zh, ocr(list), w, h, img_bytes, layout(dict)}"""
    import lmdb
    import pickle
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = {name: [] for name, _ in META_SCHEMA}
    meta_rows = []
    env = lmdb.open(str(out_dir / "mdb" / f"{stem}.mdb"), map_size=64 << 30, subdir=False)
    try:
        with env.begin(write=True) as txn:
            for r in records:
                layout_str = json.dumps(r["layout"], ensure_ascii=False)
                ocr_texts = [it["text"] for it in r["layout"]["items"] if it["label"] == "text"]
                sid = r["sample_id"]
                rec = {
                    "id": sid, "source": "breakpoint-grounding-55m",
                    "caption": r["caption"], "caption_zh": r["caption_zh"],
                    "ocr": ocr_texts, "w": r["w"], "h": r["h"],
                    "img_bytes": r["img_bytes"], "caption_ok": True,
                    "orig_id": str(r["source_index"]), "layout": layout_str,
                }
                txn.put(struct_pack_u64(sid), pickle.dumps(rec))
                meta_rows.append({
                    "sample_id": sid, "source_index": r["source_index"], "caption": r["caption"],
                    "caption_zh": r["caption_zh"], "ocr": json.dumps(ocr_texts, ensure_ascii=False),
                    "width": r["w"], "height": r["h"], "file_size": len(r["img_bytes"]),
                    "caption_ok": True, "orig_id": str(r["source_index"]), "layout": layout_str,
                })
    finally:
        env.sync()
        env.close()
    for name, _ in META_SCHEMA:
        cols[name] = [m[name] for m in meta_rows]
    arrays = [pa.array(cols[name], type=pa.type_for_alias(typ)) for name, typ in META_SCHEMA]
    table = pa.table(arrays, names=[n for n, _ in META_SCHEMA])
    p = out_dir / "parquet" / f"{stem}.parquet"
    tmp = p.with_name(f".{p.stem}.tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, p)


def struct_pack_u64(v: int) -> bytes:
    return int(v).to_bytes(8, "little", signed=False)


def process_shard(shard_path: Path, shard_index: int, root: Path, ocr, vlm, q_th: dict, l_th: dict,
                  meta_label: str) -> tuple[int, int]:
    """处理一个分片 → (合格行数, 总行数)。逐 row group 读取 (大表 take 会 offset overflow)。"""
    import pyarrow.parquet as pq

    n = pq.ParquetFile(shard_path).metadata.num_rows
    # stage1 sidecar → 质量过滤
    sidecar = load_filter_rows(root, shard_path.stem)
    if sidecar is None:
        print(f"⚠️ {shard_path.stem}: 无 stage1 sidecar, 跳过", flush=True)
        return 0, n
    pass_set = {m["index"] for m in sidecar if row_pass(m, q_th)}
    if not pass_set:
        return 0, n

    out_records = []
    t0 = time.time()
    n_done = 0
    pf = pq.ParquetFile(shard_path)
    offset = 0
    vlm_bs = 32
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg)
        gsz = tbl.num_rows
        local_idx = [i - offset for i in sorted(pass_set) if offset <= i < offset + gsz]
        offset += gsz
        if not local_idx:
            del tbl
            continue
        g_base = offset - gsz

        # 1) decode + OCR + 收集大框 crop (攒批)
        pend = []   # {idx, row, tl, w, h}
        crops = []  # (ridx, k, PIL crop, crop_box)
        for ridx, (gidx, row) in enumerate(zip(local_idx, tbl.take(local_idx).to_pylist())):
            img_bytes = row["image"]["bytes"]
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            lines = ocr_image(ocr, img)
            tl = prepare_text_lines(lines, w, h, l_th)
            pend.append(dict(idx=g_base + gidx, row=row, tl=tl, w=w, h=h))
            if vlm is not None and any(is_l for *_, is_l in tl):
                from PIL import Image
                pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                for k, c, box in collect_large_crops(pil, tl):
                    crops.append((ridx, k, c, box))
                del pil
            del img
        del tbl

        # 2) 大框 VLM 攒批推理 (一次拿 转写+bbox+style)
        vlm_out: dict[tuple, tuple] = {}
        if vlm is not None:
            for s in range(0, len(crops), vlm_bs):
                chunk = crops[s:s + vlm_bs]
                try:
                    outs = vlm.ask([c for _, _, c, _ in chunk],
                                   [LARGE_REGION_PROMPT] * len(chunk))
                except Exception as e:
                    print(f"⚠️ VLM 批推理失败 ({len(chunk)} crops): {e}", flush=True)
                    continue
                for (ridx, k, _, box), out in zip(chunk, outs):
                    j = parse_vlm_json(out)
                    if j and isinstance(j.get("bbox"), list) and len(j["bbox"]) == 4 \
                            and all(isinstance(v, (int, float)) for v in j["bbox"]):
                        vlm_out[(ridx, k)] = (j, box)

        # 3) 组装 layout + 记录
        for ridx, p in enumerate(pend):
            row = p["row"]
            summary_cap = (row.get("image_caption") or "").strip()  # 复用数据集 caption, 免每行 VLM
            out_map = {k: v for (r2, k), v in vlm_out.items() if r2 == ridx}
            layout = build_layout(row, p["tl"], p["w"], p["h"], l_th, [meta_label],
                                  summary_cap, out_map)
            img_bytes = row["image"]["bytes"]
            out_records.append(dict(sample_id=(shard_index << 20) | p["idx"], source_index=p["idx"],
                                    caption=summary_cap, caption_zh="", w=p["w"], h=p["h"],
                                    img_bytes=img_bytes, layout=layout))
            n_done += 1
            if n_done % 200 == 0:
                print(f"    {n_done}/{len(pass_set)} ({n_done/(time.time()-t0):.2f} img/s)", flush=True)
        del pend, crops, vlm_out

    out_root = root / OUT_SUBDIR
    write_shard_outputs(out_root, shard_path.stem, out_records)
    return len(out_records), n


def worker_loop(gpu_id: int, task_queue, result_queue, root_str: str, args_dict: dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    root = Path(root_str)
    log_path = root / OUT_SUBDIR / "_failed.log"

    ocr = build_ocr()
    vlm = VLM() if not args_dict["no_vlm"] else None
    print(f"[gpu{gpu_id}] OCR + VLM 就绪", flush=True)

    q_th = dict(QUALITY_DEFAULTS)
    q_th.update(min_side=args_dict["q_min_side"], blur_th=args_dict["blur_th"],
                jpeg_th=args_dict["jpeg_th"], aes_th=args_dict["aes_th"])
    l_th = dict(LAYOUT_BBOX_DEFAULTS, bbox_min_side=args_dict["bbox_min_side"],
                overlap_th=args_dict["overlap_th"], vlm_min_frac=args_dict["vlm_min_frac"])
    shards = sorted((root / "data").glob("train-*.parquet"))
    progress = Progress(root)

    while True:
        task = task_queue.get()
        if task is None:
            return
        shard = Path(task)
        shard_index = shards.index(shard)
        try:
            n_ok, n = process_shard(shard, shard_index, root, ocr, vlm, q_th, l_th, args_dict["meta"])
            progress.mark_done(shard.stem, n_ok, 0, gpu_id)
            result_queue.put(("ok", gpu_id, shard.stem, n_ok, n))
        except Exception as e:
            import traceback
            with open(log_path, "a") as f:
                f.write(f"[gpu{gpu_id} {time.strftime('%H:%M:%S')}] {shard.stem}: {e}\n{traceback.format_exc()}\n")
            result_queue.put(("fail", gpu_id, shard.stem, str(e), n := 0))


def run_full(args):
    import multiprocessing as mp

    root = Path(args.root)
    (root / OUT_SUBDIR / "parquet").mkdir(parents=True, exist_ok=True)
    (root / OUT_SUBDIR / "mdb").mkdir(parents=True, exist_ok=True)
    shards = sorted((root / "data").glob("train-*.parquet"))
    progress = Progress(root)
    todo = [s for s in shards if s.stem not in progress.done_shards()]
    print(f"📦 全量打包: {len(shards)} 分片, 待处理 {len(todo)} (已完成 {len(shards)-len(todo)})", flush=True)

    ctx = mp.get_context("spawn")
    task_queue, result_queue = ctx.Queue(), ctx.Queue()
    args_dict = dict(no_vlm=args.no_vlm, meta=args.meta,
                     q_min_side=args.q_min_side, blur_th=args.blur_th,
                     jpeg_th=args.jpeg_th, aes_th=args.aes_th,
                     bbox_min_side=args.bbox_min_side, overlap_th=args.overlap_th,
                     vlm_min_frac=args.vlm_min_frac)
    workers = [ctx.Process(target=worker_loop, args=(g, task_queue, result_queue, str(root), args_dict),
                           daemon=True) for g in range(args.num_gpus)]
    for w in workers:
        w.start()
    for s in todo:
        task_queue.put(str(s))
    for _ in workers:
        task_queue.put(None)

    n_ok_total, n_fail = 0, 0
    for _ in todo:
        status, gpu, stem, a, b = result_queue.get()
        if status == "ok":
            n_ok_total += a
            print(f"✅ [gpu{gpu}] {stem}: {a}/{b} 行合格打包 (累计 {n_ok_total})", flush=True)
        else:
            n_fail += 1
            print(f"❌ [gpu{gpu}] {stem}: {a}", flush=True)
    for w in workers:
        w.join(timeout=60)
    print(f"\n🏁 完成: {n_ok_total} 行打包, {n_fail} 分片失败 (详见 {root/OUT_SUBDIR/'_failed.log'})", flush=True)


def run_status(args):
    root = Path(args.root)
    shards = sorted((root / "data").glob("train-*.parquet"))
    snap = Progress(root).snapshot()
    total_rows = sum(v["rows"] for v in snap.values())
    print(f"进度: {len(snap)}/{len(shards)} 分片, {total_rows} 行已打包")
    for stem, v in sorted(snap.items()):
        print(f"  {stem}: {v['rows']} 行 (gpu{v['worker']})")


def run_stream(args):
    """流式模式: 与 stage1 并行, stage1 每完成一个分片 (sidecar+progress 标记) 就派给 GPU worker 打包。"""
    import multiprocessing as mp
    from filter_quality import Progress as Stage1Progress

    root = Path(args.root)
    (root / OUT_SUBDIR / "parquet").mkdir(parents=True, exist_ok=True)
    (root / OUT_SUBDIR / "mdb").mkdir(parents=True, exist_ok=True)
    shards = sorted((root / "data").glob("train-*.parquet"))
    progress = Progress(root)
    stage1 = Stage1Progress(root)

    ctx = mp.get_context("spawn")
    task_queue, result_queue = ctx.Queue(), ctx.Queue()
    args_dict = dict(no_vlm=args.no_vlm, meta=args.meta,
                     q_min_side=args.q_min_side, blur_th=args.blur_th,
                     jpeg_th=args.jpeg_th, aes_th=args.aes_th,
                     bbox_min_side=args.bbox_min_side, overlap_th=args.overlap_th,
                     vlm_min_frac=args.vlm_min_frac)
    workers = [ctx.Process(target=worker_loop, args=(g, task_queue, result_queue, str(root), args_dict),
                           daemon=True) for g in range(args.num_gpus)]
    for w in workers:
        w.start()
    print(f"🔄 流式打包: {args.num_gpus} 卡 worker 就绪, 等待 stage1 产出分片 ...", flush=True)

    queued: set[str] = set()   # 已入队/已跳过的分片
    retries: dict[str, int] = {}
    n_ok_total, n_fail = 0, 0
    idle_since = None
    while True:
        done1 = stage1.done_shards()
        done3 = progress.done_shards()
        # 新就绪分片 (stage1 完成 & stage3 未做 & 不在队列)
        ready = [s for s in shards if s.stem in done1 and s.stem not in done3 and s.stem not in queued]
        for s in ready:
            queued.add(s.stem)
            task_queue.put(str(s))
        if ready:
            print(f"📥 新就绪 {len(ready)} 片 (stage1 {len(done1)}/{len(shards)}, stage3 已完成 {len(done3)})", flush=True)

        # 非阻塞收割结果; 失败重试最多 3 次
        while True:
            try:
                status, gpu, stem, a, b = result_queue.get_nowait()
            except Exception:
                break
            if status == "ok":
                n_ok_total += a
                print(f"✅ [gpu{gpu}] {stem}: {a}/{b} 行合格打包 (累计 {n_ok_total}, stage1 进度 {len(done1)}/{len(shards)})", flush=True)
            else:
                n_fail += 1
                queued.discard(stem)
                retries[stem] = retries.get(stem, 0) + 1
                if retries[stem] > 3:
                    queued.add(stem)  # 不再入队
                    print(f"❌ [gpu{gpu}] {stem}: {a} → 重试超限, 跳过 (见 _failed.log)", flush=True)
                else:
                    print(f"❌ [gpu{gpu}] {stem}: {a} → 第 {retries[stem]} 次重试", flush=True)

        # 终止条件: stage1 全部完成且没有未完成分片
        if len(done1) >= len(shards) and all(s.stem in done3 for s in shards):
            if idle_since is None:
                idle_since = time.time()
            if time.time() - idle_since > 60:
                break
        else:
            idle_since = None
        time.sleep(30)

    for _ in workers:
        task_queue.put(None)
    for w in workers:
        w.join(timeout=120)
    print(f"\n🏁 流式完成: {n_ok_total} 行打包 (详见 {root/OUT_SUBDIR/'_failed.log'})", flush=True)


def run_merge(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(args.root)
    parts = sorted((root / OUT_SUBDIR / "parquet").glob("train-*.parquet"))
    if not parts:
        print("无分片 parquet 可合并")
        return
    tables = [pq.read_table(p) for p in parts]
    merged = pa.concat_tables(tables)
    out = root / OUT_SUBDIR / "parquet" / "meta.parquet"
    pq.write_table(merged, out)
    n_text = sum(json.loads(ocr).__len__() for ocr in merged.column("ocr").to_pylist())
    print(f"✅ 合并 {len(parts)} 片 → {out}: {merged.num_rows} 行, {n_text} 条 OCR 文本")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--meta", default="photo", help="summary.meta 标签")
    # sample
    ap.add_argument("--sample", type=int, default=8, help="sample 模式: 选文本最多的 N 张出样例")
    ap.add_argument("--scan", type=int, default=128, help="sample 模式: OCR 扫描行数")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--row-offset", type=int, default=0)
    ap.add_argument("--include-no-text", type=int, default=1, help="额外选 N 张无文本的展示 object-only")
    # 全量质量过滤阈值 (与 stage1/2 对齐)
    ap.add_argument("--q-min-side", type=int, default=QUALITY_DEFAULTS["min_side"])
    ap.add_argument("--blur-th", type=float, default=QUALITY_DEFAULTS["blur_th"])
    ap.add_argument("--jpeg-th", type=float, default=QUALITY_DEFAULTS["jpeg_th"])
    ap.add_argument("--aes-th", type=float, default=QUALITY_DEFAULTS["aes_th"])
    # layout bbox 过滤
    ap.add_argument("--bbox-min-side", type=int, default=LAYOUT_BBOX_DEFAULTS["bbox_min_side"])
    ap.add_argument("--overlap-th", type=float, default=LAYOUT_BBOX_DEFAULTS["overlap_th"])
    ap.add_argument("--vlm-min-frac", type=float, default=LAYOUT_BBOX_DEFAULTS["vlm_min_frac"])
    # 模式
    ap.add_argument("--num-gpus", type=int, default=0, help=">0 进入全量打包模式")
    ap.add_argument("--stream", action="store_true", help="流式模式: 与 stage1 并行, 分片完成即打包 (需配合 --num-gpus)")
    ap.add_argument("--no-vlm", action="store_true", help="全量模式只跑 OCR (大框用默认描述, caption 留空)")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    if args.status:
        run_status(args)
    elif args.merge:
        run_merge(args)
    elif args.stream and args.num_gpus > 0:
        run_stream(args)
    elif args.num_gpus > 0:
        run_full(args)
    else:
        run_sample(args)


if __name__ == "__main__":
    main()
