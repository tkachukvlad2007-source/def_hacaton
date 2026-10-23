#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Чесна перевірка стійкості: беремо розмічені кадри відео перевірки, псуємо їх (дощ, туман, фари,
темрява, трясіння) і рахуємо IoU маски моделі з ПРАВИЛЬНОЮ розміткою.

  python eval_stress.py --models models/road_seg.pt models/road_seg_weather.pt --videos dev_0010

Ефекти тут навмисно реалізовані інакше, ніж аугментації в weather.py (інші формули й параметри),
щоб модель не перевірялась на точно тих самих спотвореннях, на яких вчилась.
"""
import argparse
import os

import cv2
import numpy as np
import torch

from seg_video import keep_road, load_model
from train_seg import find_pairs, preprocess, video_of

W = 700  # ефекти підібрані під кадр ~700 px завширшки


def t_rain(g, rng):
    h, w = g.shape
    out = cv2.GaussianBlur(g, (5, 5), 0)
    streaks = np.zeros_like(out)
    for _ in range(250):
        x, y, ln = rng.integers(0, w), rng.integers(0, h), rng.integers(10, 35)
        cv2.line(streaks, (int(x), int(y)), (int(x + ln * 0.25), int(y + ln)), float(rng.integers(120, 255)), 1)
    out = np.maximum(out, cv2.GaussianBlur(streaks, (3, 3), 0) * 0.8)
    for _ in range(12):
        cv2.circle(out, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(8, 30)), 200.0, -1)
    return cv2.GaussianBlur(out, (3, 3), 0)


def t_fog(g, rng):
    k = np.linspace(0.85, 0.35, g.shape[0], dtype=np.float32)[:, None]
    return g * (1 - k) + 110 * k


def t_glare(g, rng):
    h, w = g.shape
    cx, cy = w * rng.uniform(0.3, 0.8), h * 0.4
    yy, xx = np.mgrid[0:h, 0:w]
    r = w * rng.uniform(0.2, 0.3)
    return g + 255 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / r ** 2))


def t_dark_noise(g, rng):
    return g * 0.35 + rng.normal(0, 18, g.shape)


def t_shake_blur(g, rng):
    k = 15
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1 / k
    return cv2.filter2D(g, -1, ker)


TESTS = {"clean": None, "rain": t_rain, "fog": t_fog, "glare": t_glare, "dark_noise": t_dark_noise,
         "motion_blur": t_shake_blur}


@torch.no_grad()
def run(model, size, dev, pairs, eff, post):
    inter = union = 0.0
    for i, (ip, mp) in enumerate(pairs):
        g = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127
        h = int(g.shape[0] * W / g.shape[1])
        g = cv2.resize(g, (W, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        m = cv2.resize(m.astype(np.uint8), (W, h), interpolation=cv2.INTER_NEAREST) > 0
        if TESTS[eff] is not None:
            g = TESTS[eff](g, np.random.default_rng(i))          # однаковий шум для всіх моделей
        g = np.clip(g, 0, 255).astype(np.uint8)
        p = torch.sigmoid(model(preprocess(g, size)[None].to(dev)))[0, 0].float().cpu().numpy()
        pm = cv2.resize(p, (W, h)) > 0.5
        if post:
            pm = keep_road(pm)
        inter += np.logical_and(pm, m).sum()
        union += np.logical_or(pm, m).sum()
    return inter / max(union, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--videos", nargs="+", default=["dev_0010"], help="відео перевірки (не бачені в навчанні!)")
    ap.add_argument("--post", action="store_true", help="з постобробкою keep_road, як у seg_video.py")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = [p for p in find_pairs(a.data) if video_of(p[0]) in a.videos]
    print(f"{len(pairs)} кадрів з {', '.join(a.videos)}" + ("  (з keep_road)" if a.post else ""))
    head = f"{'модель':32s}" + "".join(f"{e:>12s}" for e in TESTS) + f"{'середнє':>10s}"
    print(head)
    for mp in a.models:
        model, size = load_model(mp, dev)
        r = [run(model, size, dev, pairs, e, a.post) for e in TESTS]
        print(f"{os.path.basename(mp):32s}" + "".join(f"{x:12.3f}" for x in r) + f"{np.mean(r):10.3f}", flush=True)


if __name__ == "__main__":
    main()
