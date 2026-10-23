#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Авто-розмітка: модель сама обводить дорогу на кадрах, людина лише перевіряє й виправляє.

  python prelabel.py data/images --model models/quick.pt --out prelabels.json

Результат — prelabels.json з контурами (точками) для кожного кадру.
Поклади його в data/masks/_prelabels.json на ноутбуці — label_tool.py підставить ці контури
на кадри, які ще не розмічені (позначка AUTO). Правильний контур — Enter, неправильний — підсунь точки.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch

from seg_video import keep_road, load_model
from train_seg import preprocess


def mask_to_polygon(mask, max_pts=16):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.01 * mask.size:
        return []
    eps = 0.004 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True)
    while len(approx) > max_pts:          # спрощуємо, доки точок не стане зручно мало
        eps *= 1.3
        approx = cv2.approxPolyDP(c, eps, True)
    return [[float(x), float(y)] for x, y in approx[:, 0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("--model", default="models/quick.pt")
    ap.add_argument("--out", default="prelabels.json")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--skip-prefix", default="given__", help="не чіпати кадри з цим початком назви")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, size = load_model(a.model, dev)
    files = sorted(f for f in glob.glob(os.path.join(a.images, "*"))
                   if f.lower().endswith((".jpg", ".jpeg", ".png")) and not os.path.basename(f).startswith(a.skip_prefix))
    out, empty = {}, 0
    for i, f in enumerate(files):
        img = cv2.imread(f)
        h, w = img.shape[:2]
        with torch.no_grad():
            p = torch.sigmoid(model(preprocess(img, size)[None].to(dev)))[0, 0].float().cpu().numpy()
        mask = keep_road(cv2.resize(p, (w, h)) > a.thr)
        poly = mask_to_polygon(mask)
        empty += not poly
        out[os.path.splitext(os.path.basename(f))[0]] = poly
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(files)}", flush=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"Авто-контури для {len(out)} кадрів -> {a.out} (без дороги: {empty})")


if __name__ == "__main__":
    main()
