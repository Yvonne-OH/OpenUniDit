#!/usr/bin/env python3
"""关键词 -> 域名 扩池流水线
渠道: ①Bing 搜索(site:myshopify.com + 品类词, 翻页) ②GitHub 商店名录数据集
验证: /products.json 探测(Shopify) + 首页可达性
输出: new_domains_shopify.txt / new_domains_other.txt / kw_dig_report.json
"""
import argparse, json, os, re, ssl, sys, time
import urllib.request, urllib.parse, urllib.error
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
DOM_RE = re.compile(r'https?://([A-Za-z0-9][A-Za-z0-9.-]*\.[a-z]{2,})(/[^\s"\'<>]*)?', re.I)
BING_H2 = re.compile(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"', re.I)
GH_RAW = re.compile(r'https://raw\.githubusercontent\.com/[^\s"\')\]]+\.(?:csv|txt|json)', re.I)

def get(url, timeout=15, headers=None):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": UA,
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8"})
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return r.read(30*1024*1024)

def norm_dom(d):
    d = d.lower().strip()
    d = re.sub(r'^www\d?\.', '', d)
    return d if re.match(r'^[a-z0-9][a-z0-9.-]*\.[a-z]{2,}$', d) else None

CDN_SKIP = re.compile(r'^(cdn\d?|img\d*|static\d*|assets?\d*|image\d*|images?|media|pic\d*|file|files|dl|i\d|s\d|p\d)\.|alicdn|360buyimg|myqcloud|aliyuncs|bing\.com|microsoft|baidu|github|w3\.org|schema\.org|google|youtube|facebook|instagram|pinterest|tiktok|twitter|x\.com|reddit|amazon|ebay|walmart|aliexpress|taobao|tmall|jd\.com|pinduoduo|youtube|shopify\.com$|shopify\.com/|myshopify\.com$|zhuanlan|zhihu|csdn|jianshu|youtube\.com|medium\.com|linkedin', re.I)

def bing_search(kw, pages=3):
    """Bing 搜索挖域名: site:myshopify.com + 通用 query"""
    found = OrderedDict()
    qs = [f"site:myshopify.com {kw}", f'"{kw}" shopify store online']
    for q in qs:
        for pg in range(pages):
            first = pg * 10 + 1
            url = f"https://cn.bing.com/search?q={urllib.parse.quote(q)}&first={first}&count=20"
            try:
                html = get(url, timeout=12).decode("utf-8", "ignore")
            except Exception:
                continue
            for m in BING_H2.finditer(html):
                href = m.group(1)
                # bing 跳转链接还原
                if "bing.com/ck/a" in href:
                    mm = re.search(r'[?&]u=a1([^&]+)', href)
                    if mm:
                        import base64
                        pad = mm.group(1)
                        pad += "=" * (-len(pad) % 4)
                        try: href = base64.urlsafe_b64decode(pad).decode("utf-8", "ignore")
                        except Exception: continue
                dm = norm_dom(href.split("/")[2] if "://" in href else href)
                if dm and not CDN_SKIP.search(dm):
                    found[dm] = {"kw": kw, "src": "bing"}
            time.sleep(0.8)  # 对搜索 politely 限速
    return list(found.keys())

def github_store_lists(keywords):
    """GitHub 搜商店名录数据集, 抓 raw 文件解析 Domain 字段/裸链接"""
    found = OrderedDict()
    got_files = 0
    for q in ("shopify stores list", "list of shopify stores json", "shopify stores csv"):
        try:
            resp = get("https://api.github.com/search/repositories?q=" + urllib.parse.quote(q) + "&per_page=10",
                       headers={"User-Agent": UA, "Accept": "application/vnd.github+json"})
            repos = json.loads(resp).get("items", [])
        except Exception:
            continue
        for r in repos[:6]:
            if got_files >= 5: break
            try:
                tree = json.loads(get(f"https://api.github.com/repos/{r['full_name']}/git/trees/HEAD?recursive=1",
                                      headers={"User-Agent": UA, "Accept": "application/vnd.github+json"}))
            except Exception:
                continue
            for it in tree.get("tree", []):
                p = it.get("path", "")
                if not re.search(r'\.(csv|txt|json)$', p, re.I): continue
                if it.get("size", 0) > 8*1024*1024 or it.get("size", 0) < 2000: continue
                ru = f"https://raw.githubusercontent.com/{r['full_name']}/HEAD/{p}"
                try:
                    text = get(ru, timeout=25).decode("utf-8", "ignore")
                except Exception:
                    continue
                n0 = len(found)
                for m in re.finditer(r'"Domain"\s*:\s*"([^"]+)"', text):
                    dm = norm_dom(m.group(1))
                    if dm and not CDN_SKIP.search(dm):
                        found[dm] = {"kw": "(list)", "src": "github"}
                for m in DOM_RE.finditer(text):
                    dm = norm_dom(m.group(1))
                    if dm and not CDN_SKIP.search(dm):
                        found[dm] = {"kw": "(list)", "src": "github"}
                got_files += 1
                print(f"  github {r['full_name']}/{p}: +{len(found)-n0} 累计{len(found)}", flush=True)
                if got_files >= 5: break
    return list(found.keys())

def validate(domain):
    """验证域名: Shopify products.json / 首页可达"""
    rec = {"domain": domain, "shopify": False, "reachable": False, "products": 0}
    try:
        d = get(f"https://{domain}/products.json?limit=1", timeout=10)
        j = json.loads(d.decode("utf-8", "ignore"))
        if isinstance(j, dict) and "products" in j:
            rec.update(shopify=True, reachable=True)
            # 再拉一页250计满量样本
            try:
                j2 = json.loads(get(f"https://{domain}/products.json?limit=250&page=1", timeout=12).decode("utf-8", "ignore"))
                rec["products"] = sum(len(p.get("images", [])) for p in j2.get("products", []))
            except Exception: pass
            return rec
    except Exception:
        pass
    for scheme in ("https", "http"):
        try:
            d = get(f"{scheme}://{domain}/", timeout=10)
            rec["reachable"] = True
            if b"/cdn/shop/" in d or b"shopify" in d.lower():
                rec["shopify"] = True
            break
        except Exception:
            continue
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keywords", required=True, help="关键词文件, 每行一个")
    ap.add_argument("--per-kw-pages", type=int, default=3)
    ap.add_argument("--validate-limit", type=int, default=400)
    ap.add_argument("--existing", default="", help="已有域名文件(逗号分隔多个), 用于差集")
    ap.add_argument("--out-prefix", default="new_domains")
    args = ap.parse_args()

    kws = [l.strip() for l in open(args.keywords, encoding="utf-8") if l.strip()]
    print(f"关键词 {len(kws)} 个: {kws}", flush=True)

    all_d = OrderedDict()
    for kw in kws:
        ds = bing_search(kw, pages=args.per_kw_pages)
        print(f"  [{kw}] Bing 命中 {len(ds)} 域名", flush=True)
        for d in ds:
            all_d.setdefault(d, {"kw": kw, "src": "bing"})
    gh = github_store_lists(kws)
    for d in gh:
        all_d.setdefault(d, {"kw": "(list)", "src": "github"})
    print(f"渠道合计候选 {len(all_d)} 个", flush=True)

    # 差集: 去掉已有池
    existing = set()
    if args.existing:
        for fp in args.existing.split(","):
            if os.path.exists(fp):
                existing |= {l.strip().lower() for l in open(fp, encoding="utf-8", errors="ignore") if l.strip()}
    new_d = [d for d in all_d if d not in existing]
    print(f"差集后新域名 {len(new_d)} 个 (剔除已有 {len(all_d)-len(new_d)})", flush=True)

    # 验证(小批量上限)
    val_list = new_d[:args.validate_limit]
    print(f"验证前 {len(val_list)} 个...", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=20) as ex:
        futs = {ex.submit(validate, d): d for d in val_list}
        for f in as_completed(futs):
            results.append(f.result())
    shop = [r for r in results if r["shopify"]]
    other = [r for r in results if r["reachable"] and not r["shopify"]]
    dead = [r for r in results if not r["reachable"]]
    shop.sort(key=lambda r: -r["products"])

    open(f"{args.out_prefix}_shopify.txt", "w").write("\n".join(r["domain"] for r in shop))
    open(f"{args.out_prefix}_other.txt", "w").write("\n".join(r["domain"] for r in other))
    rep = {"keywords": kws, "candidates": len(all_d), "new_after_dedup": len(new_d),
           "validated": len(results), "shopify": len(shop), "reachable_other": len(other),
           "dead": len(dead), "shopify_total_products_sample": sum(r["products"] for r in shop),
           "top_shopify": [{"d": r["domain"], "imgs_p1": r["products"]} for r in shop[:15]]}
    json.dump(rep, open(f"{args.out_prefix}_report.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n=== 汇总 ===")
    print(f"Shopify 可用: {len(shop)} (首页商品图样本合计 {sum(r['products'] for r in shop)})")
    print(f"其他可达站: {len(other)}  不可达: {len(dead)}")
    print("Shopify Top10 (首页250商品含图数):")
    for r in shop[:10]:
        print(f"  {r['domain']:<42} {r['products']}")

if __name__ == "__main__":
    main()
