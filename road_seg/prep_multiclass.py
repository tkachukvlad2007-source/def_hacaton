#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Розмітка організаторів (seqN/frames + seqN/markdown, кольорові маски) -> наш мультикласовий формат.

  python prep_multiclass.py /data/seg_new/seg_dev_extended --out /data/seg_mc/train
  python prep_multiclass.py /data/seg_new/test/seg_test     --out /data/seg_mc/val

Кольори масок (RGB): рожевий (255,0,255) = дорога, блакитний (0,255,255) = рослинність,
жовтий (255,255,0) = перешкода, синій (0,0,255) = небо, чорний = не розмічено.
Індекси у вихідних масках: 0 небо, 1 дорога, 2 рослинність, 3 перешкода, 255 ігнорувати.
Оцінюються (eval.py) лише класи 1..3; небо й чорне не рахуються.
Назва файлу: <seq>__<кадр>.jpg / .png — за префіксом ділимо на навчання/перевірку.
"""
import argparse
import glob
import os
import shutil

import cv2
import numpy as np

# BGR (так читає cv2) -> індекс
COLORS_BGR = {
    (255, 0, 0): 0,      # небо (RGB 0,0,255)
    (255, 0, 255): 1,    # дорога
    (255, 255, 0): 2,    # рослинність (RGB 0,255,255)
    (0, 255, 255): 3,    # перешкода (RGB 255,255,0)
}
IGNORE = 255
CLASS_NAMES = ["sky", "road", "vegetation", "obstacle"]


def color_to_index(m_bgr):
    idx = np.full(m_bgr.shape[:2], IGNORE, np.uint8)
    for bgr, k in COLORS_BGR.items():
        idx[np.all(m_bgr == np.array(bgr, np.uint8), axis=2)] = k
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="папка з seqN/frames і seqN/markdown")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "masks"), exist_ok=True)
    seqs = sorted(d for d in glob.glob(os.path.join(a.src, "*")) if os.path.isdir(os.path.join(d, "markdown")))
    tot = np.zeros(5, np.int64)
    n = 0
    for sd in seqs:
        seq = os.path.basename(sd)
        for mp in sorted(glob.glob(os.path.join(sd, "markdown", "*.png"))):
            s = os.path.splitext(os.path.basename(mp))[0]
            ip = os.path.join(sd, "frames", s + ".jpg")
            if not os.path.exists(ip):
                continue
            idx = color_to_index(cv2.imread(mp))
            name = f"{seq}__{s}"
            shutil.copyfile(ip, os.path.join(a.out, "images", name + ".jpg"))
            cv2.imwrite(os.path.join(a.out, "masks", name + ".png"), idx)
            h = np.bincount(idx.ravel(), minlength=256)
            tot[:4] += h[:4]
            tot[4] += h[IGNORE]
            n += 1
        print(f"{seq}: готово")
    frac = tot / max(tot.sum(), 1)
    print(f"Кадрів: {n} -> {a.out}")
    print("Частки пікселів: " + ", ".join(f"{k} {frac[i] * 100:.1f}%" for i, k in enumerate(CLASS_NAMES + ["ignore"])))


if __name__ == "__main__":
    main()
