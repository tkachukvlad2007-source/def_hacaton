#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Підбір постобробки для мультикласової моделі за офіційною метрикою (копія логіки eval.py).

  python tune_post.py --models /data/seg_mc/runs/mc.pt [--tta] --root /data/seg_new/test/seg_test

Один раз рахує ймовірності для всіх кадрів (зберігає в пам'яті як uint8), далі перебирає:
  obstacle-scale — множник ймовірності «перешкода» перед argmax (<1 = обережніше),
  min-obstacle   — мінімальна площа плями «перешкода» (px), менші -> другий клас,
  smooth         — згладжування ймовірностей у часі всередині послідовності.
Друкує mIoU для всіх кадрів і для ручних (номер закінчується на 00).
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch

from predict_mc import find_frame_dirs, probs_for, remove_small_obstacles
from train_mc import load_mc_model

GT_COLORS_RGB = {1: (255, 0, 255), 2: (0, 255, 255), 3: (255, 255, 0)}


def load_gt(path):
    m = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    gt = np.zeros(m.shape[:2], np.uint8)
    for k, c in GT_COLORS_RGB.items():
        gt[np.all(m == np.array(c, np.uint8), axis=2)] = k
    return gt


def score(preds, gts, manual):
    """preds/gts: списки HxW uint8 (0 у gt = не рахується). Повертає (mIoU all, mIoU manual, per-class all)."""
    mi_all, mi_man = [], []
    pc = {1: [], 2: [], 3: []}
    for p, g, man in zip(preds, gts, manual):
        valid = g > 0
        ious = []
        for c in (1, 2, 3):
            gc = (g == c)
            pcm = (p == c) & valid
            u = np.count_nonzero(gc | pcm)
            if u:
                v = np.count_nonzero(gc & pcm) / u
                ious.append(v)
                pc[c].append(v)
        if ious:
            mi_all.append(np.mean(ious))
            if man:
                mi_man.append(np.mean(ious))
    return float(np.mean(mi_all)), float(np.mean(mi_man)) if mi_man else float("nan"), {c: float(np.mean(v)) for c, v in pc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/seg_new/test/seg_test")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--scales", type=float, nargs="*", default=[1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2])
    ap.add_argument("--areas", type=int, nargs="*", default=[0, 300, 1000, 3000])
    ap.add_argument("--smooths", type=float, nargs="*", default=[0.0, 0.5])
    ap.add_argument("--keep-tall", type=float, nargs="*", default=[0.0])
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for p in a.models:
        m, ck = load_mc_model(p, dev)
        models.append((m, tuple(ck["size"])))
        print(f"{p}: {ck['arch']} {ck['encoder']} {ck['size']} mIoU навч.перевірки {ck.get('miou', float('nan')):.4f}")

    seqs = []   # список послідовностей: (probs uint8 list, gts, manual)
    for fd in find_frame_dirs(a.root):
        P, G, M = [], [], []
        for fp in sorted(glob.glob(os.path.join(fd, "*.jpg"))):
            gp = fp.replace("/frames/", "/markdown/").replace(".jpg", ".png")
            if not os.path.exists(gp):
                continue
            pr = probs_for(models, cv2.imread(fp), dev, a.tta)
            P.append((pr * 255).astype(np.uint8))
            G.append(load_gt(gp))
            M.append(os.path.splitext(os.path.basename(fp))[0].endswith("00"))
        seqs.append((P, G, M))
        print(f"{fd}: {len(P)} кадрів")

    gts = sum((s[1] for s in seqs), [])
    manual = sum((s[2] for s in seqs), [])
    print(f"\n{'smooth':>6} {'scale':>5} {'area':>5} {'tall':>4} | {'mIoU all':>8} {'manual':>7} | road  veg   obst")
    results = []
    for sm in a.smooths:
        # накопичуємо IoU по кадрах для кожної пари (scale, area), не тримаючи float-копій у пам'яті
        acc = {(sc, ar, kt): [] for sc in a.scales for ar in a.areas for kt in a.keep_tall}   # список (ious: dict, manual)
        for P, G, M in seqs:
            prev = None
            for pr8, g, man in zip(P, G, M):
                pr = pr8.astype(np.float32) / 255
                if sm > 0 and prev is not None:
                    pr = (1 - sm) * pr + sm * prev
                prev = pr
                valid = g > 0
                for sc in a.scales:
                    q = pr.copy()
                    q[3] *= sc
                    pred0 = (q[1:].argmax(0) + 1).astype(np.uint8)
                    for ar in a.areas:
                        for kt in a.keep_tall:
                            pred = remove_small_obstacles(pred0.copy(), q, ar, kt) if ar else pred0
                            ious = {}
                            for c in (1, 2, 3):
                                gc = g == c
                                pc_ = (pred == c) & valid
                                u = np.count_nonzero(gc | pc_)
                                if u:
                                    ious[c] = np.count_nonzero(gc & pc_) / u
                            acc[(sc, ar, kt)].append((ious, man))
        for (sc, ar, kt), lst in acc.items():
            mi_all = [np.mean(list(i.values())) for i, _ in lst if i]
            mi_man = [np.mean(list(i.values())) for i, m in lst if i and m]
            pc = {c: np.mean([i[c] for i, _ in lst if c in i]) for c in (1, 2, 3)}
            s_all, s_man = float(np.mean(mi_all)), float(np.mean(mi_man)) if mi_man else float("nan")
            results.append((s_all, s_man, sm, sc, ar, kt))
            print(f"{sm:6.2f} {sc:5.2f} {ar:5d} {kt:4.2f} | {s_all:8.4f} {s_man:7.4f} | {pc[1]:.3f} {pc[2]:.3f} {pc[3]:.3f}", flush=True)
    best = max(results)
    print(f"\nНайкраще (all): mIoU {best[0]:.4f} (manual {best[1]:.4f}) при smooth={best[2]} scale={best[3]} area={best[4]} keep_tall={best[5]}")


if __name__ == "__main__":
    main()
