#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Прогноз мультикласової моделі у форматі організаторів: для кожної папки frames/ створює поруч
inference/<кадр>.png (1 канал: 1 дорога, 2 рослинність, 3 перешкода). Далі можна запускати їх eval.py.

  python predict_mc.py /data/seg_new/test/seg_test --model models/mc.pt
  python predict_mc.py /data/seg_new/test/seg_test --model models/mc.pt --tta --smooth 0.3 --min-obstacle 300

Опції:
  --tta            додатково прогноз на віддзеркаленому кадрі (усереднення ймовірностей)
  --smooth A       згладжування ймовірностей у часі між сусідніми кадрами послідовності (0 = вимкнено)
  --min-obstacle N плями «перешкода» менші за N пікселів замінюються другим за ймовірністю класом
  --models M1 M2   ансамбль: усереднити ймовірності кількох моделей
  --sky            дозволити прогноз «небо» (0) — для візуалізації; в eval.py небо не рахується
  --vis DIR        зберегти накладання маски на кадр для перегляду
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch

from train_mc import load_mc_model
from train_seg import preprocess

PALETTE_BGR = {0: (255, 0, 0), 1: (255, 0, 255), 2: (255, 255, 0), 3: (0, 255, 255)}


def find_frame_dirs(root):
    out = []
    for d, sub, _ in os.walk(root):
        if "frames" in sub:
            out.append(os.path.join(d, "frames"))
    return sorted(out)


@torch.no_grad()
def probs_for(models, img, dev, tta=False):
    """Усереднені ймовірності 4xHxW у розмірі кадру."""
    H, W = img.shape[:2]
    acc = None
    for m, size in models:
        x = preprocess(img, size)[None].to(dev)
        with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
            p = torch.softmax(m(x).float(), 1)
            if tta:
                p = 0.5 * (p + torch.flip(m(torch.flip(x, [3])).float(), [3]).softmax(1))
        p = torch.nn.functional.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)[0]
        acc = p if acc is None else acc + p
    return (acc / len(models)).cpu().numpy()


def remove_small_obstacles(pred, probs, min_area, keep_tall=0.0):
    """Плями «перешкода» менші за min_area px -> другий за ймовірністю клас.
    keep_tall: не чіпати вузькі високі плями (стовпи), якщо їх висота >= keep_tall * висота кадру."""
    if min_area <= 0:
        return pred
    ob = (pred == 3).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ob, 8)
    alt = np.where(probs[1] >= probs[2], 1, 2).astype(np.uint8)
    h_min = keep_tall * pred.shape[0]
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < min_area:
            if keep_tall > 0 and stats[k, cv2.CC_STAT_HEIGHT] >= h_min and stats[k, cv2.CC_STAT_HEIGHT] >= 3 * stats[k, cv2.CC_STAT_WIDTH]:
                continue
            sel = lab == k
            pred[sel] = alt[sel]
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--model", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--out-name", default="inference")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--smooth", type=float, default=0.0)
    ap.add_argument("--min-obstacle", type=int, default=0)
    ap.add_argument("--obstacle-scale", type=float, default=1.0, help="множник ймовірності «перешкода» (<1 = обережніше)")
    ap.add_argument("--keep-tall", type=float, default=0.0, help="лишати вузькі високі плями (стовпи) вище цієї частки кадру")
    ap.add_argument("--sky", action="store_true")
    ap.add_argument("--vis", default=None)
    a = ap.parse_args()

    paths = a.models or ([a.model] if a.model else [])
    if not paths:
        raise SystemExit("Вкажіть --model або --models")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for p in paths:
        m, ck = load_mc_model(p, dev)
        models.append((m, tuple(ck["size"])))
        print(f"{p}: {ck['arch']} {ck['encoder']} {ck['size']} mIoU перевірки {ck.get('miou', float('nan')):.4f}")

    total = 0
    for fd in find_frame_dirs(a.root):
        od = os.path.join(os.path.dirname(fd), a.out_name)
        os.makedirs(od, exist_ok=True)
        prev = None
        files = sorted(glob.glob(os.path.join(fd, "*.jpg")) + glob.glob(os.path.join(fd, "*.png")))
        for fp in files:
            img = cv2.imread(fp)
            probs = probs_for(models, img, dev, a.tta)
            if a.smooth > 0 and prev is not None and prev.shape == probs.shape:
                probs = (1 - a.smooth) * probs + a.smooth * prev
            prev = probs
            probs = probs.copy()
            probs[3] *= a.obstacle_scale
            if a.sky:
                pred = probs.argmax(0).astype(np.uint8)
            else:
                pred = (probs[1:].argmax(0) + 1).astype(np.uint8)
            pred = remove_small_obstacles(pred, probs, a.min_obstacle, a.keep_tall)
            name = os.path.splitext(os.path.basename(fp))[0] + ".png"
            cv2.imwrite(os.path.join(od, name), pred)
            if a.vis:
                vd = os.path.join(a.vis, os.path.basename(os.path.dirname(fd)))
                os.makedirs(vd, exist_ok=True)
                col = np.zeros_like(img)
                for k, c in PALETTE_BGR.items():
                    col[pred == k] = c
                cv2.imwrite(os.path.join(vd, name.replace(".png", ".jpg")),
                            cv2.addWeighted(img, 0.55, col, 0.45, 0), [cv2.IMWRITE_JPEG_QUALITY, 80])
            total += 1
        print(f"{fd}: {len(files)} кадрів -> {od}")
    print(f"Разом {total} масок")


if __name__ == "__main__":
    main()
