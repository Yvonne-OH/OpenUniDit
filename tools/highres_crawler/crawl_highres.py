#!/usr/bin/env python3
"""highres crawler — 高清图片爬取（pilot 版）

模式:
  list  : 读 urls.txt（每行一个图片URL）直接下载
  site  : 读 sites.txt（每行一个域名），爬首页发现图片URL再下载

特性: 断点续跑(progress.jsonl) / 全局+单域限速 / 20MiB上限 / magic-bytes校验
      X-Robots-Tag opt-out / 优雅退出 / 可选BOS上传(baidubce)
"""
import argparse, hashlib, io, json, os, re, signal, ssl, sys, threading, time
import urllib.request, urllib.error
from collections import defaultdict
from queue import Queue, Empty
from urllib.parse import urljoin

STOP = threading.Event()
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE
MAX_BYTES = 20 * 1024 * 1024

MAGIC = [(b"\xff\xd8\xff", "jpg"), (b"\x89PNG\r\n\x1a\n", "png"), (b"GIF8", "gif"),
         (b"RIFF", "webp"), (b"BM", "bmp")]
IMGEXT = re.compile(r'\.(jpg|jpeg|png|webp|gif|bmp)(\?|&|$)', re.I)
IMGHTML = re.compile(r'''(?:src|data-src|data-original|href)["']?\s*[:=]\s*["']([^"']+\.(?:jpg|jpeg|png|webp|gif)[^"']*)["']''', re.I)
SRCSET = re.compile(r'''srcset=["']([^"']+)["']''', re.I)
OGIMG = re.compile(r'''og:image["'][^>]*content=["']([^"']+)["']|content=["']([^"']+)["'][^>]*og:image''', re.I)
SKIP_PAT = re.compile(r'\.(svg|ico|css|js)(\?|$)|logo|icon|avatar|sprite|badge|\.gif(\?|$)', re.I)
PAGE_LINK = re.compile(r'''(?:href|data-url|data-href)["']?\s*[:=]\s*["']([^"']+)["']''', re.I)
GOODPAGE = re.compile(r'(product|collection|goods|item|detail|category|blog|post|gallery|photo|shop|list|archive)', re.I)
LOCS = re.compile(r'<loc>\s*([^<\s]+)\s*</loc>', re.I)
WM_DOMAIN = re.compile(r'gettyimages|shutterstock|alamy|dreamstime|istockphoto|123rf|bigstockphoto|depositphotos|stockfresh|fotolia|agefotostock|pond5|colourbox|imago-images|photocase|scanstock|photographyblog|watermark', re.I)

# ---------- 图像质量筛选阈值 ----------
Q_MIN_BPP = 0.8       # bpp=bytes*8/(w*h), 低于=过度压缩糊图
Q_BLUR_TH = 30.0      # Laplacian方差(512降采样), 低于=模糊
Q_BRIGHT_LO = 12.0    # 亮度过低(近黑)
Q_BRIGHT_HI = 243.0   # 亮度过高(近白)
Q_CONTRAST_TH = 8.0   # 灰度std过低(无细节)
Q_GRAY_TH = 10.0      # RGB通道饱和度均值过低=黑白图
Q_MIN_SIDE = 512      # 短边小于512直接拒绝

# URL 级小图预筛: URL 里明确带 <512 尺寸参数的不入队, 省下载带宽
SMALL_URL = re.compile(
    r'[?&/](?:width|w|resize_w|img_width|maxwidth)=(\d{1,4})/?'      # width=300 / w=200
    r'|x-oss-process=[^&]*?w[_/](\d{1,4})'                          # 阿里云oss resize
    r'|/resize/(\d{1,4})x'                                          # /resize/200x
    r'|imageview2/[^&]*?/w/(\d{1,4})[/&?]'                          # 七牛 imageView2
    r'|[/_.](\d{1,4})x(\d{1,4})[_.@]?(?:jpg|jpeg|png|webp)',        # 200x200.jpg 命名
    re.I)
def is_small_url(u):
    for m in SMALL_URL.finditer(u):
        for g in m.groups():
            if g and int(g) < 512:
                return True
    return False

STATS = defaultdict(int); SLOCK = threading.Lock()
def bump(k, n=1):
    with SLOCK: STATS[k] += n

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

import threading as _th
_TLS = _th.local()

def get_session():
    """每线程一个 keep-alive Session, 复用 TCP/TLS 连接"""
    s = getattr(_TLS, "sess", None)
    if s is None:
        try:
            import requests, urllib3
            from requests.adapters import HTTPAdapter
            urllib3.disable_warnings()
            s = requests.Session()
            ad = HTTPAdapter(pool_connections=4, pool_maxsize=12)
            s.mount("https://", ad); s.mount("http://", ad)
            s.headers.update({"User-Agent": UA, "Referer": "https://www.google.com/",
                              "Accept": "image/*,*/*;q=0.8"})
        except Exception:
            s = False  # requests 不可用标记
        _TLS.sess = s
    return s or None

def fetch(url, timeout=15):
    """返回 (bytes, headers) 或抛异常; 优先连接池复用"""
    s = get_session()
    if s is None:  # urllib 兜底
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": "https://www.google.com/",
                                                   "Accept": "image/*,*/*;q=0.8"})
        with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
            xrt = r.headers.get("X-Robots-Tag", "") or ""
            if "noimageindex" in xrt.lower():
                raise PermissionError("x-robots-tag-noimageindex")
            data = r.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                raise ValueError("too-large")
            return data, r.headers
    r = s.get(url, timeout=timeout, stream=True, verify=False)
    xrt = r.headers.get("X-Robots-Tag", "") or ""
    if "noimageindex" in xrt.lower():
        r.close(); raise PermissionError("x-robots-tag-noimageindex")
    if r.status_code >= 400:
        r.close()
        raise urllib.error.HTTPError(url, r.status_code, "http error", r.headers, None)
    r.raw.decode_content = True
    data = r.raw.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        r.close(); raise ValueError("too-large")
    hdrs = r.headers
    r.close()  # 连接归还连接池
    return data, hdrs

def fetch2(url, timeout=15, tries=3):
    """带瞬态错误重试: 连接重置/超时/DNS抖动 自动重试; HTTP语义错误(404/403等)不重试"""
    last = None
    for k in range(tries):
        try:
            return fetch(url, timeout=timeout)
        except (urllib.error.HTTPError, PermissionError, ValueError):
            raise  # HTTP 语义错误重试无意义
        except Exception as e:
            last = e
            time.sleep(0.6 * (k + 1))
    raise last

def sniff(data):
    for magic, ext in MAGIC:
        if data.startswith(magic):
            if ext == "webp" and data[8:12] != b"WEBP": continue
            return ext
    return None

# ---------- 每域名限速信号量 ----------
# 共享CDN(所有Shopify站共用cdn.shopify.com等)不是源站, 放宽到8; 普通源站3
CDN_HOST = re.compile(r'alicdn|360buyimg|myqcloud|aliyuncs|cdn\.shopify|wp\.com|gtimg|qhimg|hdslb|sicdn|taobaocdn|byteimg|toutiaoimg|imgur|cloudfront|akamaized|fastly', re.I)
DLOCK = threading.Lock()
DSEMS = {}
def dsem(domain):
    with DLOCK:
        if domain not in DSEMS:
            n = 8 if CDN_HOST.search(domain) else 3
            DSEMS[domain] = threading.BoundedSemaphore(n)
        return DSEMS[domain]

class Crawler:
    def __init__(self, out, workers=16, limit=0, seconds=0, bos=None, pages_per_site=40, wm=None):
        self.pages_per_site = pages_per_site
        self.wm = wm
        self.t0 = time.time()
        self.out = out; self.imgdir = os.path.join(out, "images")
        os.makedirs(self.imgdir, exist_ok=True)
        self.limit = limit; self.deadline = time.time() + seconds if seconds else 0
        self.bos = bos
        self.done = set()
        self.pj = os.path.join(out, "progress.jsonl")
        if os.path.exists(self.pj):
            with open(self.pj) as f:
                for line in f:
                    try: self.done.add(json.loads(line)["url"])
                    except Exception: pass
            log(f"断点: 已有 {len(self.done)} 条记录")
        self.plock = threading.Lock()
        self.q = Queue()
        self.seen = set()
        self.slock = threading.Lock()
        self.bos_q = Queue()
        self.bos_up = None

    def record(self, url, status, extra=None):
        rec = {"t": time.time(), "url": url, "status": status}
        if extra: rec.update(extra)
        with self.plock:
            with open(self.pj, "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.done.add(url)

    def record_bos(self, url, path, ext):
        with self.plock:
            with open(self.pj, "a") as f:
                f.write(json.dumps({"t": time.time(), "url": url, "status": "bos_uploaded",
                                    "file": path, "ext": ext}) + "\n")

    def enqueue(self, url):
        with self.slock:
            if url in self.seen or url in self.done: return
            self.seen.add(url)
        if is_small_url(url):
            bump("url_small"); return
        self.q.put(url)

    def below_limits(self):
        if self.limit and STATS.get("saved", 0) >= self.limit: return False
        if self.deadline and time.time() > self.deadline: return False
        return True

    def quality_check(self, data):
        """CPU 质量筛选: 返回 (拒因或None, 指标dict)。解析/依赖失败不拦图保吞吐"""
        try:
            from PIL import Image
            import numpy as np
            im = Image.open(io.BytesIO(data)).convert("RGB")
            w, h = im.size
            if min(w, h) < Q_MIN_SIDE:
                return "too_small", {}
            bpp = len(data) * 8.0 / (w * h)
            sm = im.copy(); sm.thumbnail((512, 512))
            a = np.asarray(sm, dtype=np.float32)
            g = a.mean(axis=2)
            lap = 4 * g - np.roll(g, 1, 0) - np.roll(g, -1, 0) - np.roll(g, 1, 1) - np.roll(g, -1, 1)
            blur = float(lap[1:-1, 1:-1].var())
            bright = float(g.mean())
            contrast = float(g.std())
            sat = float((a.max(axis=2) - a.min(axis=2)).mean())
            m = {"bpp": round(bpp, 2), "blur": round(blur, 1), "bright": round(bright, 1),
                 "contrast": round(contrast, 1), "sat": round(sat, 1)}
            if bpp < Q_MIN_BPP: return "low_bpp", m
            if blur < Q_BLUR_TH: return "blurry", m
            if bright < Q_BRIGHT_LO or bright > Q_BRIGHT_HI: return "bad_brightness", m
            if contrast < Q_CONTRAST_TH: return "low_contrast", m
            if sat < Q_GRAY_TH: return "grayscale", m
            return None, m
        except Exception:
            return None, {}

    def download(self, url):
        if not self.below_limits(): return
        try:
            dom = url.split("/")[2]
        except Exception:
            return
        if WM_DOMAIN.search(dom):
            bump("wm_domain"); self.record(url, "wm_domain"); return
        with dsem(dom):
            if not self.below_limits(): return
            data = None
            for attempt in (1, 2):  # 千万级: 失败重试1次
                try:
                    data, _hdrs = fetch(url)
                    break
                except PermissionError as e:
                    bump("optout"); self.record(url, "optout"); return
                except urllib.error.HTTPError as e:
                    bump(f"http_{e.code}"); self.record(url, f"http_{e.code}"); return
                except Exception as e:
                    if attempt == 1:
                        time.sleep(0.5)
                        continue
                    bump("net_fail"); self.record(url, "fail", {"err": type(e).__name__}); return
        ext = sniff(data)
        if not ext:
            bump("bad_magic"); self.record(url, "bad_magic"); return
        qreason, qm = self.quality_check(data)
        if qreason:
            bump(f"q_{qreason}"); self.record(url, f"q_{qreason}", qm); return
        if self.wm and self.wm_check(data):
            bump("wm_skip"); self.record(url, "watermarked"); return
        h = hashlib.sha1(data).hexdigest()[:16]
        # 目录分级: {域名}/{h前2位}/{h次2位}/, 单目录文件数控制在几百内
        dom_dir = os.path.join(self.imgdir, re.sub(r"[^a-z0-9.-]", "_", dom.lower())[:60], h[:2], h[2:4])
        os.makedirs(dom_dir, exist_ok=True)
        path = os.path.join(dom_dir, f"{h}.{ext}")
        if not os.path.exists(path):
            with open(path, "wb") as f: f.write(data)
        bump("saved"); bump("bytes", len(data))
        self.record(url, "saved", {"file": path, "bytes": len(data), **qm})
        if self.bos:
            self.bos_q.put((url, path, ext, 0))

    BOS_MAX_TRIES = 5

    def bos_upload_once(self, batch, cl):
        """单轮逐张上传: 返回仍失败的(未达重试上限)列表"""
        rest = []
        for url, path, ext, tries in batch:
            bn = os.path.basename(path)
            key = f"{self.bos['prefix']}/{ext}/{bn[:2]}/{bn}"
            try:
                cl.put_object_from_file(self.bos["bucket"], key, path)
                os.remove(path)
                self.record_bos(url, key, ext)
            except Exception as e:
                if tries + 1 < self.BOS_MAX_TRIES and os.path.exists(path):
                    rest.append((url, path, ext, tries + 1))
                else:
                    bump("bos_fail")
                    log(f"BOS 最终失败({type(e).__name__}) {bn[:40]}")
        return rest

    def bos_upload(self, batch, cl):
        """运行期: 失败项重新入队, 由上传线程稍后重试"""
        rest = self.bos_upload_once(batch, cl)
        if rest:
            bump("bos_retry", len(rest))
            time.sleep(2)
            for it in rest:
                self.bos_q.put(it)

    def bos_worker(self):
        """并行上传线程(4个): 各自独立BosClient, 攒批25张"""
        cl = bos_client(self.bos["cred"])
        while not (STOP.is_set() and self.bos_q.empty()):
            try:
                item = self.bos_q.get(timeout=5)
            except Empty:
                continue
            batch = [item]
            while len(batch) < 25:
                try: batch.append(self.bos_q.get_nowait())
                except Empty: break
            self.bos_upload(batch, cl)

    def bos_flush(self):
        """退出兜底: 主线程清空剩余队列"""
        if not self.bos: return
        batch = []
        while True:
            try: batch.append(self.bos_q.get_nowait())
            except Empty: break
        if not batch: return
        cl = bos_client(self.bos["cred"])
        pending = batch
        for rnd in range(self.BOS_MAX_TRIES):
            rest = self.bos_upload_once(pending, cl)
            if not rest:
                return
            pending = rest
            time.sleep(3)
        bump("bos_fail", len(pending))

    def worker(self):
        while not STOP.is_set():
            try: url = self.q.get(timeout=1.0)
            except Empty:
                if self.deadline and time.time() > self.deadline: break
                continue
            try: self.download(url)
            except Exception as e:
                bump("internal_err"); log(f"internal {type(e).__name__}: {e}")
            finally: self.q.task_done()

    def site_discover(self, domain):
        """深爬站点: 首页 + sitemap + Shopify products.json + 商品/列表页跟进"""
        # 域名变体回退: 裸域 -> www. (不少站只挂在 www 上), scheme https -> http
        cand_hosts = [domain.lower()]
        if not cand_hosts[0].startswith("www."):
            cand_hosts.append("www." + cand_hosts[0])
        html = None
        base = None
        for h in cand_hosts:
            for scheme in ("https", "http"):
                try:
                    data, _ = fetch(f"{scheme}://{h}/")
                    html = data.decode("utf-8", "ignore")
                    base = f"{scheme}://{h}/"
                    domain = h
                    break
                except Exception:
                    continue
            if html: break
        if not html:
            bump("site_fail"); return
        bump("site_ok")
        domain_clean = domain.lower()

        def emit(u):
            u = u.strip().replace("\\u002F", "/").replace("&amp;", "&")
            if u.startswith("//"): u = "https:" + u
            u = urljoin(base, u).split(" ")[0].split("#")[0]
            return u

        # 1) 首页图片
        for u in self.extract_imgs(html, base):
            self.enqueue(u)
        n_first = self.q.qsize()

        # 2) sitemap.xml: 支持 index 套娃(最多10个子sitemap), loc 图片直接下载, 页面进跟进队列
        pages = []
        seen_pages = set()
        sitemap_done = False
        sm_urls = [f"https://{domain_clean}/sitemap.xml", f"https://{domain_clean}/sitemap_index.xml"]
        sm_locs = []
        checked_sm = set()
        for sm_url in sm_urls[:]:
            if len(sm_urls) >= 12: break
            try:
                sm, _ = fetch2(sm_url, timeout=10)
            except Exception:
                continue
            sitemap_done = True
            locs = [emit(m.group(1)) for m in LOCS.finditer(sm.decode("utf-8", "ignore"))]
            for u in locs[:2000]:
                lu = u.lower().split("?")[0]
                if lu.endswith(".xml") and u not in checked_sm:
                    checked_sm.add(u)
                    if len(sm_urls) < 12: sm_urls.append(u)   # sitemap index 套娃
                    continue
                if IMGEXT.search(lu):
                    sm_locs.append(u)
                elif domain_clean in u.lower() and GOODPAGE.search(u) and u not in seen_pages:
                    seen_pages.add(u); pages.append(u)

        for u in sm_locs:
            self.enqueue(u)
        log(f"site {domain}: sitemap 图片 loc {len(sm_locs)}")

        # 3) Shopify products.json(每页250商品, 最多120页)
        shopify = "/cdn/shop/" in html or "shopify" in html.lower() or "shopify.com" in html.lower()
        if shopify:
            for pg in range(1, 121):
                try:
                    d, _ = fetch2(f"https://{domain_clean}/products.json?limit=250&page={pg}", timeout=12)
                    prods = json.loads(d.decode("utf-8", "ignore")).get("products", [])
                except Exception:
                    break
                if not prods: break
                for p in prods:
                    for im in p.get("images", []):
                        src = im.get("src") or ""
                        if src and not SKIP_PAT.search(src.lower()): self.enqueue(emit(src))
                if pg % 5 == 0: log(f"site {domain}: products.json page={pg} 队列={self.q.qsize()}")

        # 4) 页面跟进(商品/列表/博客), 每站最多 pages_max 页
        cnt = 0
        while pages and cnt < self.pages_per_site:
            u = pages.pop(0)
            cnt += 1
            try:
                d, _ = fetch2(u, timeout=10)
            except Exception:
                continue
            h = d.decode("utf-8", "ignore")
            for im in self.extract_imgs(h, base):
                self.enqueue(im)
            for m in PAGE_LINK.finditer(h):
                pu = emit(m.group(1))
                if domain_clean in pu.lower() and GOODPAGE.search(pu) and pu not in seen_pages:
                    seen_pages.add(pu); pages.append(pu)
            if cnt % 10 == 0: log(f"site {domain}: 已跟进{cnt}页 队列={self.q.qsize()}")
        log(f"site {domain}: 首页图{n_first} sitemap={sitemap_done} shopify={shopify} 跟进{cnt}页 队列={self.q.qsize()}")

    def extract_imgs(self, html, base="https://x/"):
        found = set()
        for m in IMGHTML.finditer(html): found.add(m.group(1))
        for m in SRCSET.finditer(html):
            for p in m.group(1).split(","):
                p = p.strip().split(" ")[0]
                if IMGEXT.search(p): found.add(p)
        for m in OGIMG.finditer(html):
            u = m.group(1) or m.group(2)
            if u: found.add(u)
        out = []
        for u in found:
            u = u.strip().replace("\\u002F", "/")
            if u.startswith("//"): u = "https:" + u
            u = urljoin(base, u).split(" ")[0].split("#")[0]
            if IMGEXT.search(u.split("?")[0]) and not SKIP_PAT.search(u.lower()):
                out.append(u)
        return out

    def site_worker(self, sites_q):
        while not STOP.is_set():
            try: d = sites_q.get(timeout=1.0)
            except Empty:
                return
            try: self.site_discover(d)
            except Exception as e:
                bump("site_err"); log(f"site {d} 异常 {type(e).__name__}: {str(e)[:60]}")
            finally: sites_q.task_done()

    def run(self, urls=None, sites=None):
        nwork = int(os.environ.get("CRAWL_WORKERS", "32"))
        log(f"下载worker={nwork}, 每域并发: 源站3/共享CDN8")
        threads = [threading.Thread(target=self.worker, daemon=True) for _ in range(nwork)]
        for t in threads: t.start()
        if self.bos:
            self.bos_up = [threading.Thread(target=self.bos_worker, daemon=True) for _ in range(4)]
            for t in self.bos_up: t.start()
        site_threads = []
        sites_q = Queue()
        if sites:
            for d in sites: sites_q.put(d)
            site_threads = [threading.Thread(target=self.site_worker, args=(sites_q,), daemon=True) for _ in range(8)]
            for t in site_threads: t.start()
        if urls:
            for i, u in enumerate(urls):
                if not self.below_limits(): break
                self.enqueue(u)
                if i % 2000 == 0 and i: log(f"list 入队 {i}")
        # 等待队列消化
        while not STOP.is_set():
            qsize = self.q.qsize()
            sites_alive = any(t.is_alive() for t in site_threads) if site_threads else False
            if self.limit and STATS.get("saved", 0) >= self.limit:
                log(f"达到数量上限 {self.limit}"); break
            if self.deadline and time.time() > self.deadline:
                log("达到时间上限"); break
            if qsize == 0 and not sites_alive:
                time.sleep(2)
                if self.q.qsize() == 0: break
            time.sleep(2)
        STOP.set()
        for t in threads: t.join(timeout=30)
        if self.bos_up:
            for t in self.bos_up: t.join(timeout=180)
        self.bos_flush()
        summary = dict(stats=dict(STATS), elapsed=int(time.time() - self.t0),
                       done_records=len(self.done))
        with open(os.path.join(self.out, "summary.json"), "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        log(f"结束: {json.dumps(dict(STATS))}")

    def wm_check(self, data):
        """ONNX 水印分类(SigLIP2, 输入[-1,1]): True=水印图(丢弃)"""
        try:
            from PIL import Image
            import numpy as np
            im = Image.open(io.BytesIO(data)).convert("RGB").resize((224, 224))
            x = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
            x = np.transpose(x, (2, 0, 1))[None]
            out = self.wm["sess"].run(None, {self.wm["inname"]: x})[0]
            z = out[0].astype(np.float64)
            e = np.exp(z - z.max())
            p_wm = float(e[1] / e.sum())
            return p_wm > 0.6
        except Exception:
            return False  # 推理失败不拦图, 保吞吐

def load_wm(path):
    if not path or not os.path.exists(path): return None
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        log(f"水印模型已加载: {path}")
        return {"sess": sess, "inname": sess.get_inputs()[0].name}
    except Exception as e:
        log(f"水印模型加载失败({type(e).__name__}), 跳过检测")
        return None

_TLS_BOS = _th.local()
def bos_client(cred):
    """每线程独立 BosClient(线程安全)"""
    cl = getattr(_TLS_BOS, "cl", None)
    if cl is None:
        import sys as _s
        _vdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pylibs")
        if os.path.isdir(_vdir) and _vdir not in _s.path: _s.path.insert(0, _vdir)
        from baidubce.auth.bce_credentials import BceCredentials
        from baidubce.bce_client_configuration import BceClientConfiguration
        from baidubce.services.bos.bos_client import BosClient
        conf = BceClientConfiguration(credentials=BceCredentials(cred["ak"], cred["sk"]),
                                      endpoint=cred.get("endpoint", "https://bj.bcebos.com"))
        conf.path_style_enable = True
        cl = BosClient(conf)
        _TLS_BOS.cl = cl
    return cl

def load_bos(prefix):
    cred = "/home/jovyan/pfs/users/zhanqian.wu/root/.bos_center_cred"
    if not os.path.exists(cred): return None
    try:
        c = json.load(open(cred))
        log("BOS 可用(4并行上传线程)")
        return {"bucket": c["bucket"], "prefix": prefix, "cred": c}
    except Exception as e:
        log(f"BOS 不可用({type(e).__name__}), 仅落 PFS")
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["list", "site", "both"], default="list")
    ap.add_argument("--urls", default=""); ap.add_argument("--sites", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seconds", type=int, default=0)
    ap.add_argument("--sites-max", type=int, default=0)
    ap.add_argument("--pages-per-site", type=int, default=40)
    ap.add_argument("--bos-prefix", default="users/zhanqian.wu/highres_images/pilot")
    ap.add_argument("--wm-model", default="")
    ap.add_argument("--no-bos", action="store_true")
    args = ap.parse_args()

    urls = [l.strip() for l in open(args.urls) if l.strip()] if args.urls and args.mode in ("list", "both") else []
    sites = [l.strip() for l in open(args.sites) if l.strip()] if args.sites and args.mode in ("site", "both") else []
    if args.sites_max: sites = sites[:args.sites_max]

    # 公网连通性探测
    try:
        fetch("https://www.baidu.com/favicon.ico", timeout=10)
        log("公网连通: OK")
    except Exception as e:
        log(f"公网连通失败: {type(e).__name__} — 继续尝试")

    crawler = Crawler(args.out, limit=args.limit, seconds=args.seconds, pages_per_site=args.pages_per_site,
                      bos=None if args.no_bos else load_bos(args.bos_prefix),
                      wm=load_wm(args.wm_model))
    def on_term(sig, fm):
        log("收到停止信号, 优雅退出"); STOP.set()
    signal.signal(signal.SIGTERM, on_term); signal.signal(signal.SIGINT, on_term)
    crawler.run(urls=urls, sites=sites)
    rc = 0 if STATS.get("saved", 0) > 0 else 1
    sys.exit(rc)

if __name__ == "__main__":
    main()
