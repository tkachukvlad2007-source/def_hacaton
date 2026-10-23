#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Збирає з відео дані для підбору керування: геометрію дороги (з моделі) і те, як насправді
повертав водій (кут повороту камери між кадрами з оптичного потоку по верхній частині кадру).

  python collect_drive.py ../../data/dev_0001.mp4 ... --out /data/road_seg/drive_data.npz
"""
import argparse

import cv2
import numpy as np
import torch

from seg_video import confidence, keep_road, load_model, road_geometry
from train_seg import preprocess


def yaw_shift(prev, cur):
    """Зсув далекого фону по горизонталі між кадрами, частка ширини кадру (>0 = машина повертає праворуч)."""
    (dx, _), resp = cv2.phaseCorrelate(prev, cur)
    return -dx / prev.shape[1], resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--model", default="models/road_seg_weather.pt")
    ap.add_argument("--out", default="/data/road_seg/drive_data.npz")
    ap.add_argument("--step", type=int, default=1)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, size = load_model(a.model, dev)
    rows = []
    for vi, src in enumerate(a.videos):
        cap = cv2.VideoCapture(src)
        w0, h0 = int(cap.get(3)), int(cap.get(4))
        w, h = 640, int(640 * h0 / w0) // 2 * 2
        prev_g, prev_mask, prob, i = None, None, None, 0
        win = None
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            fr = cv2.resize(fr, (w, h))
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            top = np.float32(gray[: int(h * 0.45)])
            if win is None:
                win = cv2.createHanningWindow(top.shape[::-1], cv2.CV_32F)
            with torch.no_grad():
                p = torch.sigmoid(model(preprocess(fr, size)[None].to(dev)))[0, 0].float().cpu().numpy()
            prob = p if prob is None else 0.5 * prob + 0.5 * p
            mask = keep_road(cv2.resize(prob, (w, h)) > 0.5)
            g = road_geometry(mask)
            c = confidence(cv2.resize(p, (w, h)), prev_mask, mask, gray)
            prev_mask = mask
            yaw, resp = yaw_shift(prev_g * win, top * win) if prev_g is not None else (0.0, 0.0)
            prev_g = top
            if g is None:
                rows.append([vi, i, yaw, resp, c] + [np.nan] * 6)
            else:
                yb, yt = g["rows"].max(), g["rows"].min()
                # центр дороги на 6 відстанях попереду (частка видимої дороги), відносно центру кадру
                cs = [(np.polyval(g["coef"], yb - f * (yb - yt)) - w / 2) / w for f in (0.0, 0.2, 0.4, 0.6, 0.8)]
                rows.append([vi, i, yaw, resp, c, g["offset"]] + cs)
            i += 1
        print(src, i, "кадрів", flush=True)
    np.savez(a.out, data=np.array(rows, np.float32), videos=np.array(a.videos))
    print("->", a.out)


if __name__ == "__main__":
    main()
