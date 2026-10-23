#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Навчання сегментації дороги (один клас: дорога від краю до краю).

  python train_seg.py --data data --out models/road_seg.pt
  python train_seg.py --data data --val-videos dev_0003 dev_0007 --epochs 40

Очікує:
  data/images/<відео>__<кадр>.jpg   — кадри (extract_frames.py)
  data/masks/<відео>__<кадр>.png    — маски (label_tool.py), 255 = дорога

Перевірка йде на окремих ВІДЕО (не кадрах) — інакше оцінка буде завищена.
Працює і на GPU (Colab), і на процесорі (повільно).
"""
import argparse
import glob
import math
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def video_of(path):
    return os.path.basename(path).split("__")[0]


def preprocess(img, size):
    """BGR або сірий кадр -> тензор 3xHxW. Кадр завжди переводимо в сірий (нічна ІЧ-камера)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if (g.shape[1], g.shape[0]) != tuple(size):
        g = cv2.resize(g, tuple(size), interpolation=cv2.INTER_AREA)
    x = np.repeat(g[..., None].astype(np.float32) / 255.0, 3, axis=2)
    x = (x - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def build_model(encoder="mobilenet_v2", pretrained=True):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                    in_channels=3, classes=1)


def augment(g, m, rng, weather=0.0):
    """Аугментація під нічну ІЧ-камеру на трясучій платформі.
    weather — імовірність додати погану погоду (засвітка, дощ, туман, темрява; див. weather.py)."""
    h, w = g.shape
    if rng.random() < 0.5:
        g, m = g[:, ::-1], m[:, ::-1]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-7, 7), rng.uniform(0.9, 1.15))
    M[:, 2] += (rng.uniform(-0.06, 0.06) * w, rng.uniform(-0.06, 0.06) * h)
    g = cv2.warpAffine(np.ascontiguousarray(g), M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    m = cv2.warpAffine(np.ascontiguousarray(m), M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    f = g.astype(np.float32) / 255.0
    f = np.clip((f - 0.5) * rng.uniform(0.7, 1.3) + 0.5 + rng.uniform(-0.15, 0.15), 0, 1)
    f = f ** rng.uniform(0.7, 1.5)
    if rng.random() < 0.5:   # пляма ІЧ-підсвітки або віньєтка
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w * rng.uniform(0.3, 0.7), h * rng.uniform(0.5, 1.0)
        r = ((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2
        f = np.clip(f + rng.uniform(-0.25, 0.4) * np.exp(-r / rng.uniform(0.02, 0.12)), 0, 1)
    if rng.random() < 0.25:
        k = int(rng.choice([3, 5, 7]))
        f = cv2.GaussianBlur(f, (k, k), 0)
    if rng.random() < 0.25:  # змазування від трясіння
        k = int(rng.choice([5, 9, 13]))
        ker = np.zeros((k, k), np.float32)
        if rng.random() < 0.5:
            ker[k // 2, :] = 1.0
        else:
            ker[:, k // 2] = 1.0
        f = cv2.filter2D(f, -1, ker / k)
    if rng.random() < weather:
        from weather import add_weather
        f = add_weather(f, rng)
    f = np.clip(f + rng.normal(0, rng.uniform(0, 0.05), f.shape), 0, 1)
    return (f * 255).astype(np.uint8), m


class RoadDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, size, train, weather=0.0):
        self.pairs, self.size, self.train, self.weather = pairs, tuple(size), train, weather

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        g = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        g = cv2.resize(g, self.size, interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, self.size, interpolation=cv2.INTER_NEAREST)
        if self.train:
            g, m = augment(g, m, np.random.default_rng(), self.weather)
        return preprocess(g, self.size), torch.from_numpy((m > 127).astype(np.float32))[None]


def find_pairs(data):
    pairs = []
    for mp in sorted(glob.glob(os.path.join(data, "masks", "*.png"))):
        s = os.path.splitext(os.path.basename(mp))[0]
        for ext in (".jpg", ".jpeg", ".png"):
            ip = os.path.join(data, "images", s + ext)
            if os.path.exists(ip):
                pairs.append((ip, mp))
                break
    return pairs


def dice_loss(logits, y):
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3))
    return (1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)).mean()


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    inter = union = 0.0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        p = (torch.sigmoid(model(x)) > 0.5).float()
        inter += (p * y).sum().item()
        union += ((p + y) > 0).float().sum().item()
    return inter / max(union, 1.0)


def main():
    ap = argparse.ArgumentParser(description="Навчання сегментації дороги")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="models/road_seg.pt")
    ap.add_argument("--val-videos", nargs="*", default=None, help="відео лише для перевірки (назви без .mp4)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, nargs=2, default=[512, 288], help="ширина висота (кратні 32)")
    ap.add_argument("--encoder", default="mobilenet_v2")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--weather", type=float, default=0.0,
                    help="імовірність аугментації поганою погодою (0.4 — стійкіша до засвітки й дощу)")
    a = ap.parse_args()

    if a.size[0] % 32 or a.size[1] % 32:
        raise SystemExit("--size має бути кратним 32, напр. 512 288")
    pairs = find_pairs(a.data)
    if len(pairs) < 10:
        raise SystemExit(f"Знайдено лише {len(pairs)} розмічених кадрів у {a.data}. Спершу розмітьте кадри.")
    vids = sorted({video_of(p[0]) for p in pairs})
    if a.val_videos:
        val_v = set(os.path.splitext(v)[0] for v in a.val_videos)
    elif len(vids) >= 2:
        val_v = set(vids[-max(1, round(len(vids) * 0.2)):])
    else:
        val_v = set()
    tr = [p for p in pairs if video_of(p[0]) not in val_v]
    va = [p for p in pairs if video_of(p[0]) in val_v]
    if not va or not tr:
        print("УВАГА: усі кадри з одного відео — перевірка буде на випадкових кадрах того ж відео "
              "і покаже завищену якість. Розмітьте кадри хоча б з 2-3 відео.")
        random.Random(0).shuffle(pairs)
        cut = max(1, int(len(pairs) * 0.15))
        va, tr = pairs[:cut], pairs[cut:]
        val_v = {"(випадкові кадри)"}
    print(f"Відео: {', '.join(vids)}")
    print(f"Навчання: {len(tr)} кадрів | перевірка: {len(va)} кадрів з {', '.join(sorted(val_v))}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Пристрій: {dev}" + ("" if dev == "cuda" else "  (без GPU буде повільно)"))
    dl_tr = torch.utils.data.DataLoader(RoadDataset(tr, a.size, True, a.weather), batch_size=a.batch, shuffle=True,
                                        num_workers=a.workers, drop_last=len(tr) > a.batch)
    dl_va = torch.utils.data.DataLoader(RoadDataset(va, a.size, False), batch_size=a.batch, num_workers=a.workers)
    model = build_model(a.encoder, pretrained=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    best = -1.0
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(x)
            n += len(x)
        sched.step()
        iou = evaluate(model, dl_va, dev)
        mark = ""
        if iou > best:
            best = iou
            torch.save({"state": model.state_dict(), "encoder": a.encoder, "size": list(a.size),
                        "iou": best, "val_videos": sorted(val_v), "weather": a.weather}, a.out)
            mark = "  <- збережено"
        print(f"епоха {ep:3d}/{a.epochs}  loss {tot / max(n, 1):.4f}  IoU перевірки {iou:.3f}  "
              f"({time.time() - t0:.0f} с){mark}", flush=True)
    print(f"\nГотово. Найкращий IoU на перевірці: {best:.3f}. Модель: {a.out}")
    print("IoU — частка збігу маски моделі з вашою розміткою: 0.85+ дуже добре, 0.7 терпимо, <0.6 погано.")


if __name__ == "__main__":
    main()
