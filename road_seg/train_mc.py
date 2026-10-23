#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Навчання мультикласової сегментації: небо / дорога / рослинність / перешкода.

  python train_mc.py --train /data/seg_mc/train --val /data/seg_mc/val --out models/mc_unet_mbv2.pt
  python train_mc.py --arch Unet --encoder efficientnet-b4 --size 704 352 --batch 4 --epochs 30

Дані: <папка>/images/<seq>__<кадр>.jpg + <папка>/masks/<seq>__<кадр>.png
(індекси 0 небо, 1 дорога, 2 рослинність, 3 перешкода, 255 ігнорувати; див. prep_multiclass.py).

Метрика перевірки = як в eval.py організаторів: для кожного кадру IoU класів 1..3 (лише на розмічених
пікселях), середнє по класах, що є в кадрі або в прогнозі, потім середнє по кадрах. Небо в прогноз
не йде: argmax беремо лише серед класів 1..3 (на оцінюваних пікселях небо завжди помилка).
Зберігається найкраща епоха за mIoU перевірки.
"""
import argparse
import glob
import math
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from train_seg import preprocess, augment

IGNORE = 255
NOT_ROAD = 254   # часткова розмітка: точно не дорога, але невідомо що саме (бінарні маски label_tool)
CLASS_NAMES = ["sky", "road", "vegetation", "obstacle"]
EVAL_CLASSES = (1, 2, 3)


def build_mc_model(arch="Unet", encoder="mobilenet_v2", pretrained=True, classes=4):
    import segmentation_models_pytorch as smp
    kw = dict(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
              in_channels=3, classes=classes)
    return getattr(smp, arch)(**kw)


def load_mc_model(path, dev="cuda"):
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = build_mc_model(ck["arch"], ck["encoder"], pretrained=False, classes=ck.get("classes", 4))
    m.load_state_dict(ck["state"])
    return m.to(dev).eval(), ck


def find_pairs(folder):
    pairs = []
    for mp in sorted(glob.glob(os.path.join(folder, "masks", "*.png"))):
        s = os.path.splitext(os.path.basename(mp))[0]
        ip = os.path.join(folder, "images", s + ".jpg")
        if os.path.exists(ip):
            pairs.append((ip, mp))
    return pairs


def find_binary_pairs(folder):
    """Бінарні маски (255 = дорога) з розмітника; кадри організаторів given__ пропускаємо (вони вже є як seq0_old)."""
    pairs = []
    for mp in sorted(glob.glob(os.path.join(folder, "masks", "*.png"))):
        s = os.path.splitext(os.path.basename(mp))[0]
        if s.startswith("given__"):
            continue
        for ext in (".jpg", ".jpeg", ".png"):
            ip = os.path.join(folder, "images", s + ext)
            if os.path.exists(ip):
                pairs.append((ip, mp))
                break
    return pairs


class MCDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, size, train, weather=0.0, binary=False):
        self.pairs, self.size, self.train, self.weather, self.binary = pairs, tuple(size), train, weather, binary

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        g = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if self.binary:
            m = np.where(m > 127, 1, NOT_ROAD).astype(np.uint8)
        g = cv2.resize(g, self.size, interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, self.size, interpolation=cv2.INTER_NEAREST)
        if self.train:
            g, m = augment(g, m, np.random.default_rng(), self.weather)
        return preprocess(g, self.size), torch.from_numpy(np.ascontiguousarray(m).astype(np.int64))


def dice_loss_mc(logits, y, classes=EVAL_CLASSES):
    """М'який Dice по оцінюваних класах. 255 не враховується; 254 (точно не дорога) рахується лише для класу «дорога»."""
    p = torch.softmax(logits.float(), 1)
    full = (y <= 3).float()
    road_valid = ((y <= 3) | (y == NOT_ROAD)).float()
    loss = 0.0
    for c in classes:
        valid = road_valid if c == 1 else full
        t = (y == c).float() * valid
        pc = p[:, c] * valid
        inter = (pc * t).sum((1, 2))
        loss = loss + (1 - (2 * inter + 1) / (pc.sum((1, 2)) + t.sum((1, 2)) + 1)).mean()
    return loss / len(classes)


def seg_loss(logits, y, ce_weight=None):
    """CE на повністю розмічених пікселях + «не дорога» на частково розмічених + Dice."""
    y_ce = y.clone()
    partial = y == NOT_ROAD
    y_ce[partial] = IGNORE
    loss = F.cross_entropy(logits, y_ce, weight=ce_weight, ignore_index=IGNORE)
    if partial.any():
        lg = logits.float()
        log_not_road = torch.logsumexp(lg[:, [0, 2, 3]], 1) - torch.logsumexp(lg, 1)
        loss = loss - log_not_road[partial].mean()
    return loss + dice_loss_mc(logits, y)


def predict_classes(logits, use_sky=False):
    """Індекс класу на піксель. За замовчуванням небо не прогнозується (argmax серед 1..3)."""
    if use_sky:
        return logits.argmax(1)
    return logits[:, 1:].argmax(1) + 1


def image_miou(pred, gt):
    """pred, gt: HxW numpy (int). Повертає (mIoU кадру або nan, dict IoU по класах)."""
    valid = (gt == 1) | (gt == 2) | (gt == 3)
    ious = {}
    for c in EVAL_CLASSES:
        g = (gt == c) & valid
        p = (pred == c) & valid
        u = np.count_nonzero(g | p)
        if u:
            ious[c] = np.count_nonzero(g & p) / u
    return (float(np.mean(list(ious.values()))) if ious else float("nan")), ious


@torch.no_grad()
def evaluate(model, loader, dev, use_sky=False):
    model.eval()
    mious, per_class = [], {c: [] for c in EVAL_CLASSES}
    for x, y in loader:
        with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
            logits = model(x.to(dev))
        pred = predict_classes(logits.float(), use_sky).cpu().numpy()
        y = y.numpy()
        for i in range(len(y)):
            mi, ious = image_miou(pred[i], y[i])
            if not math.isnan(mi):
                mious.append(mi)
            for c, v in ious.items():
                per_class[c].append(v)
    return (float(np.mean(mious)) if mious else 0.0), {CLASS_NAMES[c]: float(np.mean(v)) if v else float("nan")
                                                        for c, v in per_class.items()}


def main():
    ap = argparse.ArgumentParser(description="Навчання мультикласової сегментації")
    ap.add_argument("--train", nargs="+", default=["/data/seg_mc/train"])
    ap.add_argument("--val", nargs="+", default=["/data/seg_mc/val"])
    ap.add_argument("--extra-binary", nargs="*", default=[],
                    help="папки з бінарними масками дороги (255 = дорога) — часткова розмітка «дорога / не дорога»")
    ap.add_argument("--out", default="models/mc.pt")
    ap.add_argument("--arch", default="Unet", help="Unet, UnetPlusPlus, FPN, UPerNet, Segformer, DeepLabV3Plus")
    ap.add_argument("--encoder", default="mobilenet_v2")
    ap.add_argument("--size", type=int, nargs=2, default=[704, 352], help="ширина висота (кратні 32)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder-lr-mult", type=float, default=0.5, help="менший LR для попередньо навченого енкодера")
    ap.add_argument("--weather", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ce-weight", type=float, nargs=4, default=None, help="ваги CE для класів 0..3")
    ap.add_argument("--init", default=None, help="почати з цієї моделі (донавчання)")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()

    tr = sum((find_pairs(d) for d in a.train), [])
    va = sum((find_pairs(d) for d in a.val), [])
    trb = sum((find_binary_pairs(d) for d in a.extra_binary), [])
    seqs_tr = sorted({os.path.basename(p[0]).split("__")[0] for p in tr})
    seqs_va = sorted({os.path.basename(p[0]).split("__")[0] for p in va})
    print(f"Навчання: {len(tr)} кадрів ({', '.join(seqs_tr)}) + {len(trb)} бінарних | "
          f"перевірка: {len(va)} кадрів ({', '.join(seqs_va)})")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds_tr = MCDataset(tr, a.size, True, a.weather)
    if trb:
        ds_tr = torch.utils.data.ConcatDataset([ds_tr, MCDataset(trb, a.size, True, a.weather, binary=True)])
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=a.batch, shuffle=True,
                                        num_workers=a.workers, drop_last=True, pin_memory=True, persistent_workers=True)
    dl_va = torch.utils.data.DataLoader(MCDataset(va, a.size, False), batch_size=a.batch, num_workers=a.workers)

    model = build_mc_model(a.arch, a.encoder, pretrained=a.init is None).to(dev)
    if a.init:
        ck = torch.load(a.init, map_location=dev, weights_only=False)
        model.load_state_dict(ck["state"])
        print(f"Ініціалізовано з {a.init} (mIoU {ck.get('miou', float('nan')):.3f})")
    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    rest = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr * a.encoder_lr_mult},
                             {"params": rest, "lr": a.lr}], weight_decay=1e-4)
    steps = a.epochs * len(dl_tr)
    warm = min(200, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    w = torch.tensor(a.ce_weight, device=dev) if a.ce_weight else None
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    logf = open(a.log, "a") if a.log else None

    def log(s):
        print(s, flush=True)
        if logf:
            logf.write(s + "\n")
            logf.flush()

    log(f"# {a.arch} {a.encoder} size={a.size} batch={a.batch} lr={a.lr} weather={a.weather} epochs={a.epochs}")
    best = -1.0
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
                logits = model(x)
                loss = seg_loss(logits, y, w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(x)
            n += len(x)
        miou, pc = evaluate(model, dl_va, dev)
        mark = ""
        if miou > best:
            best = miou
            torch.save({"state": model.state_dict(), "arch": a.arch, "encoder": a.encoder, "size": list(a.size),
                        "classes": 4, "class_names": CLASS_NAMES, "miou": best, "per_class": pc,
                        "val_seqs": seqs_va, "train_seqs": seqs_tr, "extra_binary": len(trb)}, a.out)
            mark = "  <- збережено"
        log(f"епоха {ep:3d}/{a.epochs}  loss {tot / max(n, 1):.4f}  mIoU {miou:.4f}  "
            f"(дорога {pc['road']:.3f} росл {pc['vegetation']:.3f} переш {pc['obstacle']:.3f})  "
            f"{time.time() - t0:.0f} с{mark}")
    log(f"Готово. Найкращий mIoU перевірки: {best:.4f}. Модель: {a.out}")


if __name__ == "__main__":
    main()
