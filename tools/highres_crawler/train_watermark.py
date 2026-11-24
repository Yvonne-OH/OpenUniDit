#!/usr/bin/env python3
"""水印分类器训练: BOS 样本(域名即标签) → MobileNetV3-small CPU → ONNX
正样本: gettyimages 等图库站(必有水印); 负样本: 1688/Shopify/wp.com 商品图(干净)
"""
import argparse, io, json, os, random, time
import numpy as np
from PIL import Image

OUT = "/home/jovyan/pfs/users/zhanqian.wu/exp/highres_crawl/wm_train"
TOOL = "/home/jovyan/pfs/users/zhanqian.wu/code/model_exp/tools/highres_crawler"
PROGRESS_DIRS = ["/home/jovyan/pfs/users/zhanqian.wu/exp/highres_crawl/fleet-20261123-1405-s%d" % i for i in range(8)] + \
                ["/home/jovyan/pfs/users/zhanqian.wu/exp/highres_crawl/fleet-20261123-1458-s%d" % i for i in range(3)]
POS_DOM = ["gettyimages.com"]
NEG_DOM = ["cbu01.alicdn.com", "img.alicdn.com", "cdn.shopify.com", "cdn.store-assets.com",
           "i0.wp.com", "i1.wp.com", "i2.wp.com", "nimg.ws.126.net"]
N_POS, N_NEG = 2400, 5000
EPOCHS, BATCH, LR = 6, 48, 1e-3

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def collect_keys():
    """从 progress.jsonl 收集 (boskey, label)"""
    import json
    pos, neg = [], []
    for d in PROGRESS_DIRS:
        p = os.path.join(d, "progress.jsonl")
        if not os.path.exists(p): continue
        with open(p) as f:
            for line in f:
                if "\"bos_uploaded\"" not in line: continue
                try: r = json.loads(line)
                except: continue
                u = r.get("url", ""); k = r.get("file", "")
                if "://" not in u or not k: continue
                dom = u.split("/")[2]
                if any(x in dom for x in POS_DOM): pos.append(k)
                elif any(x in dom for x in NEG_DOM): neg.append(k)
    random.seed(1); random.shuffle(pos); random.shuffle(neg)
    return pos[:N_POS], neg[:N_NEG]

def fetch_img(bos, bucket, key):
    try:
        data = bos.get_object(bucket, key).read()
        im = Image.open(io.BytesIO(data)).convert("RGB").resize((224, 224))
        return np.asarray(im, dtype=np.float32)
    except Exception:
        return None

def main():
    os.makedirs(OUT, exist_ok=True)
    log("收集带标签的 BOS key...")
    posk, negk = collect_keys()
    log(f"正样本 key {len(posk)}, 负样本 key {len(negk)}")
    if len(posk) < 300:
        log("正样本不足, 退出"); open(os.path.join(OUT, "exit_code"), "w").write("1"); return

    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    torch.set_num_threads(14)

    from baidubce.bce_client_configuration import BceClientConfiguration
    from baidubce.auth.bce_credentials import BceCredentials
    from baidubce.services.bos.bos_client import BosClient
    c = json.load(open("/home/jovyan/pfs/users/zhanqian.wu/root/.bos_center_cred"))
    conf = BceClientConfiguration(credentials=BceCredentials(c["ak"], c["sk"]), endpoint=c["endpoint"])
    conf.path_style_enable = True
    bos = BosClient(conf)
    bucket = c["bucket"]

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    X, Y = [], []
    for label, keys in ((1, posk), (0, negk)):
        n_ok = 0
        for i, k in enumerate(keys):
            a = fetch_img(bos, bucket, k)
            if a is None: continue
            x = (a / 255.0 - MEAN) / STD
            X.append(np.transpose(x, (2, 0, 1))); Y.append(label); n_ok += 1
            if i % 800 == 0: log(f"label={label} 下载 {i}/{len(keys)} ok={n_ok}")
        log(f"label={label} 成功 {n_ok}")
    X = np.stack(X); Y = np.array(Y, dtype=np.int64)
    log(f"数据集: {X.shape[0]} 张")

    idx = np.random.RandomState(7).permutation(len(X))
    n_val = int(len(X) * 0.15)
    va, tr = idx[:n_val], idx[n_val:]

    class DS(Dataset):
        def __init__(self, ids): self.ids = ids
        def __len__(self): return len(self.ids)
        def __getitem__(self, i):
            j = self.ids[i]
            return torch.from_numpy(X[j]), torch.tensor(Y[j])

    import torchvision
    model = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()
    tl = DataLoader(DS(tr), batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True)
    vl = DataLoader(DS(va), batch_size=BATCH, num_workers=2)

    best_acc = 0.0
    for ep in range(EPOCHS):
        model.train(); tot = corr = 0
        for xb, yb in tl:
            opt.zero_grad()
            out = model(xb); loss = lossf(out, yb)
            loss.backward(); opt.step()
            corr += (out.argmax(1) == yb).sum().item(); tot += len(yb)
        model.eval(); vc = vt = 0
        with torch.no_grad():
            for xb, yb in vl:
                out = model(xb); vc += (out.argmax(1) == yb).sum().item(); vt += len(yb)
        acc = vc / max(vt, 1)
        best_acc = max(best_acc, acc)
        log(f"epoch {ep+1}: train_acc={corr/max(tot,1):.3f} val_acc={acc:.3f}")

    # ONNX 导出
    dummy = torch.randn(1, 3, 224, 224)
    onnx_path = os.path.join(TOOL, "wm_model.onnx")
    torch.onnx.export(model.cpu(), dummy, onnx_path, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=13)
    size = os.path.getsize(onnx_path) / 1e6
    log(f"ONNX 导出: {onnx_path} ({size:.1f}MB) best_val_acc={best_acc:.3f}")
    json.dump({"val_acc": best_acc, "pos": int((Y==1).sum()), "neg": int((Y==0).sum()),
               "onnx": onnx_path, "ts": time.strftime("%FT%TZ", time.gmtime())},
              open(os.path.join(OUT, "metrics.json"), "w"))
    open(os.path.join(OUT, "exit_code"), "w").write("0")
    log("完成")

if __name__ == "__main__":
    main()
