#!/usr/bin/env python3
"""构建 benchmark/prompts/ 下的四个评测 prompt 集 (jsonl).

统一格式: {"id": str, "benchmark": str, "prompt": str, "meta": dict}

来源说明:
- geneval  : GenEval 官方 553 条, github.com/djghosh13/geneval prompts/evaluation_metadata.jsonl
             (任务描述中的 zfj3214/GenEval 为已失效镜像, 官方仓库即 djghosh13/geneval)
- hpsv2    : HPS v2 官方 3200 条 (800 x 4 类), HF 数据集 zhwang/HPDv2 的 benchmark/{anime,concept-art,paintings,photo}.json
             (与 pip 包 hpsv2 的 utils.hps_prompt_files 一致, 为官方分发渠道)
- heim     : HEIM (NeurIPS 2023, arXiv 2311.04287) 官方数据仓库 stanford-crfm/heim 已失效 (404),
             故按论文所述场景取其底层数据集的官方源构建代表性子集:
             PartiPrompts (google-research/parti PartiPrompts.tsv, HEIM 用于 alignment/reasoning/knowledge 评测)
             按 Category 分层采样 300 条, 并在 meta 中标注截取方式. 属于 HEIM 场景数据的官方子集.
- qwen_image_bench : QwenLM/Qwen-Image 官方仓库未随附 benchmark prompts (已核对 repo 全部文件),
             按其技术报告 (arXiv 2508.02324) 所述评测维度人工构造 100 条, 非官方.

用法:
    python build_prompts.py --src-dir /tmp/bench_src   # 从本地已下载的源构建 (推荐)
    python build_prompts.py --download                 # 现场下载源再构建
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import urllib.request
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent

GENEVAL_URL = "https://raw.githubusercontent.com/djghosh13/geneval/main/prompts/evaluation_metadata.jsonl"
HPSV2_BASE = "https://hf-mirror.com/datasets/zhwang/HPDv2/resolve/main/benchmark"
HPSV2_FILES = {"anime": "anime", "concept-art": "concept-art", "paintings": "paintings", "photo": "photo"}
PARTI_URL = "https://raw.githubusercontent.com/google-research/parti/main/PartiPrompts.tsv"

HEIM_SUBSET_SIZE = 300  # HEIM 子集条数 (按 Category 分层采样)

# ── Qwen-Image Bench: 按技术报告维度构造的 100 条 (非官方) ──────────────────────────
# 维度: 文字渲染(英/中)、多人交互、人物肖像、动物、食物、风景、建筑、艺术风格、科幻/产品
QWEN_BENCH_TEMPLATES = {
    "text_en": [
        'A neon sign above a retro diner that says "OPEN 24 HOURS" in glowing red letters.',
        'A handwritten chalkboard menu outside a French cafe reading "Croissants - Fresh Daily".',
        'A white sign that says "Very Deep Learning" standing at the edge of the Grand Canyon.',
        'A stack of hardcover books with the title "Quantum Mechanics Made Simple" on the spine.',
        'A movie poster titled "MIDNIGHT EXPRESS" with a sleek black car under city lights.',
        'A warehouse wall painted with large letters reading "INNOVATE OR STAGNATE".',
        'A wooden shop door with a small plaque engraved "EST. 1897".',
        'A birthday cake with icing that spells "Happy 30th Birthday Amy".',
        'A stadium scoreboard displaying "FINAL: 3 - 2" in bright LED digits.',
        'A vintage airline poster with the slogan "Fly the Skies, Feel the Freedom".',
    ],
    "text_zh": [
        '一块木质招牌，上面用毛笔字写着"老北京炸酱面"。',
        '春节场景：红色的春联贴在门框两侧，写着"岁岁平安"。',
        '一家茶馆的牌匾上写着"一品香茶庄"四个金色大字。',
        '地铁站的电子屏显示"下一站：人民广场"。',
        '书法卷轴上写着"宁静致远"，旁边放着一支毛笔。',
        '一张中文报纸头版，标题是"城市早餐文化调查"。',
        '校园黑板报写着"开学快乐"，配上彩色粉笔画。',
        '便利店的玻璃门贴着"24小时营业"的标志。',
        '中式婚礼的红色横幅写着"百年好合"。',
        '博物馆展厅墙上刻着"千年文明"四个大字。',
    ],
    "multi_person": [
        "Three friends laughing together while sharing pizza at a crowded outdoor food market.",
        "A jazz quartet performing on a dimly lit stage, each musician absorbed in playing.",
        "Two children building a sandcastle on the beach while their parents watch from a picnic blanket.",
        "A busy ramen shop with a chef behind the counter and five customers eating at the bar.",
        "A handshake between a suited businessman and an astronaut in front of a rocket launchpad.",
        "Four students around a library table, surrounded by open textbooks and laptops, studying for exams.",
        "A family of four flying kites on a windy autumn hillside.",
        "Two street artists collaborating on a large colorful mural, paint splattered everywhere.",
        "A wedding ceremony with the couple exchanging vows under a flower arch, guests seated around them.",
        "Five cyclists racing down a cobblestone street during a city race.",
    ],
    "portrait": [
        "Close-up portrait of an elderly fisherman with deep wrinkles and a weathered yellow raincoat, soft overcast light.",
        "A portrait photo of a young ballerina in a white tutu, soft window light, shallow depth of field.",
        "Studio portrait of a confident woman with short silver hair, teal background, dramatic rim lighting.",
        "Close-up of a chef with flour-dusted cheeks and a warm smile, rustic kitchen bokeh behind.",
        "A golden-hour portrait of a horse rider in a field, wind-blown hair, cinematic backlight.",
        "Portrait of a violinist mid-performance, eyes closed, warm concert-hall lighting.",
        "A macro portrait of a beekeeper wearing a net veil, bees hovering around the frame.",
        "Moody portrait of a lighthouse keeper by a rain-streaked window, lantern glow.",
        "A joyful toddler with chocolate-smeared face gripping a cookie, natural kitchen light.",
        "Portrait of a samurai in full armor standing in falling cherry blossoms, dramatic side light.",
    ],
    "animal": [
        "A red fox curled up asleep on a mossy forest floor, dappled sunlight through the trees.",
        "A corgi wearing tiny sunglasses riding in a bicycle basket down a sunny street.",
        "A macro shot of a jumping spider on a dew-covered leaf at dawn.",
        "A humpback whale breaching the ocean surface at sunset, water spray frozen mid-air.",
        "A flock of flamingos standing in a milky-blue lagoon, one preening its feathers.",
        "A snow leopard prowling along a rocky Himalayan ridge in light snowfall.",
        "A squirrel holding a tiny umbrella under a mushroom in a whimsical forest scene.",
        "An octopus curled around a glass jar on the seabed, rays of light through clear water.",
        "A barn owl in mid-flight over a wheat field at golden hour, wings fully spread.",
        "Two otters holding hands while floating on their backs in a calm river.",
    ],
    "food": [
        "A towering double-stack burger with melted cheese dripping down, on a wooden board with fries.",
        "A delicate slice of matcha roll cake on a ceramic plate, dusted with powdered sugar.",
        "A steaming bowl of hand-pulled noodles with a soft-boiled egg, chopsticks lifting strands.",
        "A rustic sourdough loaf torn open to reveal an airy crumb, flour dust in the air.",
        "An overhead flat lay of a Mediterranean brunch spread: hummus, olives, pita, and fresh herbs.",
        "A chocolate lava cake cut open with molten center flowing onto a white plate.",
        "A sizzling Korean BBQ table filled with banchan dishes and marinated short ribs.",
        "A street-food stall scene: takoyaki balls being flipped with two picks, bonito flakes falling.",
        "A crystal-cut glass of iced amber cocktail garnished with orange peel, backlit by bar lights.",
        "A perfect croissant torn in half showing honeycomb layers, steam rising gently.",
    ],
    "landscape": [
        "A misty mountain valley at dawn, layered peaks fading into golden fog.",
        "Northern lights dancing over a snow-covered Icelandic church with a red roof.",
        "A turquoise alpine lake mirroring jagged snow-capped peaks under a clear sky.",
        "Rolling lavender fields in Provence at sunset, a lone farmhouse on the horizon.",
        "A dramatic lightning storm over a vast desert mesa, long-exposure style.",
        "Cherry blossom trees lining a calm river in Kyoto, petals floating on the water.",
        "A bioluminescent beach at night, glowing blue waves washing onto dark sand.",
        "Autumn forest road covered in fallen maple leaves, morning fog between the trees.",
        "A towering waterfall pouring into a lush green canyon with a rainbow in the mist.",
        "The Milky Way arching over ancient sand dunes in the Namib desert.",
    ],
    "architecture": [
        "A brutalist concrete library interior with dramatic skylights and floating staircases.",
        "A traditional Japanese tea house surrounded by raked gravel garden and maple trees.",
        "Futuristic curved glass skyscrapers reflecting a pink and orange dusk sky.",
        "A cozy bookshop interior with floor-to-ceiling shelves, spiral staircase, and warm lamps.",
        "An ancient Roman amphitheater at blue hour, warm torches glowing along the stone tiers.",
        "A minimalist white villa with an infinity pool overlooking the Amalfi coast.",
        "A grand cathedral nave with ribbed vaults and stained-glass light pooling on the floor.",
        "A treehouse hotel room cantilevered over a rainforest canopy, warm interior lights.",
        "A restoration-hardware style loft with exposed brick, steel beams, and industrial pendant lights.",
        "A Moroccan riad courtyard with zellige tiles, arched doorways, and a central fountain.",
    ],
    "art_style": [
        "Dogs playing poker, Baroque period art with dramatic chiaroscuro.",
        "A watercolor painting of a rainy Paris street with soft bleeding edges.",
        "A pixel art scene of a knight facing a dragon outside a 16-bit castle.",
        "An oil painting of a sunflower field in the style of Van Gogh with thick impasto strokes.",
        "A ukiyo-e woodblock print of a great wave with Mount Fuji and fishing boats.",
        "A cubist portrait of a jazz musician with geometric fractured planes.",
        "An art nouveau poster of a peacock woman with ornamental floral borders.",
        "A minimalist line drawing of two hands reaching, single continuous stroke, on cream paper.",
        "A low-poly 3D render of a mountain village at sunset, faceted geometry.",
        "A chalk pastel drawing of a hummingbird hovering over trumpet flowers.",
    ],
    "scifi_product": [
        "A levitating transparent smartphone showing a holographic city map, product shot on a gradient backdrop.",
        "A retro-futuristic 1960s space station interior with orange upholstery and CRT monitors.",
        "A sleek electric hypercar parked in a rain-slicked neon-lit Tokyo alley at night.",
        "A pair of augmented-reality glasses floating above a white pedestal, studio lighting.",
        "An astronaut tending a greenhouse of leafy vegetables inside a Mars habitat module.",
        "A steampunk brass diving helmet with glowing portholes and pressure gauges.",
        "A smart coffee machine brewing with visible internal gears and steam, cutaway view.",
        "A giant orbital ring station above a blue planet, solar panels glinting.",
        "A mechanical wristwatch with exposed tourbillon and meteorite dial, macro product shot.",
        "A courtyard of service robots charging under neon arches in a cyberpunk depot.",
    ],
}


def _http_get(url: str) -> bytes:
    """简单 HTTP GET (禁用代理环境差异, 30s 超时)."""
    req = urllib.request.Request(url, headers={"User-Agent": "model-exp-benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def build_geneval(src_dir: Path | None, out: Path) -> int:
    """GenEval 官方 553 条: 保留原始对象/计数元数据."""
    path = (src_dir / "geneval" / "prompts" / "evaluation_metadata.jsonl") if src_dir else None
    lines: list[dict] = []
    if path and path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                lines.append(json.loads(line))
    else:
        for line in _http_get(GENEVAL_URL).decode("utf-8").splitlines():
            if line.strip():
                lines.append(json.loads(line))
    assert len(lines) == 553, f"GenEval 应为 553 条, 实际 {len(lines)}"
    with out.open("w", encoding="utf-8") as f:
        for i, item in enumerate(lines):
            f.write(json.dumps({
                "id": f"geneval-{i:04d}",
                "benchmark": "geneval",
                "prompt": item["prompt"],
                "meta": {"tag": item.get("tag"), "include": item.get("include")},
            }, ensure_ascii=False) + "\n")
    return len(lines)


def build_hpsv2(src_dir: Path | None, out: Path) -> int:
    """HPS v2 官方 3200 条: 4 类各 800."""
    total = 0
    with out.open("w", encoding="utf-8") as f:
        for cat, fname in HPSV2_FILES.items():
            path = (src_dir / "hpsv2_prompts" / f"{fname}.json") if src_dir else None
            prompts = json.loads(path.read_text(encoding="utf-8")) if path and path.is_file() \
                else json.loads(_http_get(f"{HPSV2_BASE}/{fname}.json"))
            assert len(prompts) == 800, f"{cat} 应为 800 条, 实际 {len(prompts)}"
            for p in prompts:
                f.write(json.dumps({
                    "id": f"hpsv2-{cat}-{total:05d}",
                    "benchmark": "hpsv2",
                    "prompt": p,
                    "meta": {"category": cat},
                }, ensure_ascii=False) + "\n")
                total += 1
    assert total == 3200, f"HPSv2 应为 3200 条, 实际 {total}"
    return total


def build_heim(src_dir: Path | None, out: Path) -> int:
    """HEIM 子集: 从 PartiPrompts (HEIM 所用场景数据源) 按 Category 分层采样."""
    path = src_dir / "parti" / "PartiPrompts.tsv" if src_dir else None
    raw = path.read_text(encoding="utf-8") if path and path.is_file() else _http_get(PARTI_URL).decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(raw), delimiter="\t"))
    assert len(rows) >= 1600, f"PartiPrompts 应有 1632 条, 实际 {len(rows)}"

    # 按 Category 分层采样, 保证覆盖 P2 的全部大类别
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault((r.get("Category") or "uncategorized").strip(), []).append(r)
    rng = random.Random(42)  # 固定种子, 采样可复现
    picked: list[dict] = []
    cats = sorted(by_cat)
    per_cat = max(HEIM_SUBSET_SIZE // len(cats), 1)
    for cat in cats:
        pool = by_cat[cat]
        rng.shuffle(pool)
        picked.extend(pool[:per_cat])
    picked = picked[:HEIM_SUBSET_SIZE]

    with out.open("w", encoding="utf-8") as f:
        for i, r in enumerate(picked):
            f.write(json.dumps({
                "id": f"heim-{i:04d}",
                "benchmark": "heim",
                "prompt": r["Prompt"].strip(),
                "meta": {
                    "source": "PartiPrompts (HEIM scenario subset)",
                    "category": (r.get("Category") or "").strip(),
                    "challenge": (r.get("Challenge") or "").strip(),
                    "note": "HEIM 官方 repo (stanford-crfm/heim) 已失效; 本子集取自 HEIM 论文 "
                            "(arXiv 2311.04287) 所用场景数据源 PartiPrompts 的分层采样",
                },
            }, ensure_ascii=False) + "\n")
    return len(picked)


def build_qwen_bench(out: Path) -> int:
    """Qwen-Image Bench: 技术报告维度构造 100 条 (非官方)."""
    with out.open("w", encoding="utf-8") as f:
        i = 0
        for dim, prompts in QWEN_BENCH_TEMPLATES.items():
            for p in prompts:
                f.write(json.dumps({
                    "id": f"qwenbench-{i:04d}",
                    "benchmark": "qwen_image_bench",
                    "prompt": p,
                    "meta": {"dimension": dim,
                             "note": "非官方: 按 Qwen-Image 技术报告 (arXiv 2508.02324) 评测维度构造"},
                }, ensure_ascii=False) + "\n")
                i += 1
    return i


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-dir", type=Path, default=None,
                    help="本地源目录 (含 geneval/ hpsv2_prompts/ parti/); 缺省直接从网络拉取")
    args = ap.parse_args()
    src = args.src_dir

    results = {
        "geneval_prompts.jsonl": build_geneval(src, PROMPTS_DIR / "geneval_prompts.jsonl"),
        "hpsv2_prompts.jsonl": build_hpsv2(src, PROMPTS_DIR / "hpsv2_prompts.jsonl"),
        "heim_prompts.jsonl": build_heim(src, PROMPTS_DIR / "heim_prompts.jsonl"),
        "qwen_image_bench_prompts.jsonl": build_qwen_bench(PROMPTS_DIR / "qwen_image_bench_prompts.jsonl"),
    }
    for name, n in results.items():
        print(f"{name}: {n} 条")


if __name__ == "__main__":
    main()
