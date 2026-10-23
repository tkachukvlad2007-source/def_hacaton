#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Конвертує готову розмітку організаторів (seg/frames + seg/markdown) у наш формат.

  python convert_given.py ПАПКА_SEG --data data --name given

ПАПКА_SEG — папка, де лежать frames/ і markdown/ (можна вказати і верхню, скрипт знайде сам).
Кольори в масках: рожевий = дорога, синій = небо, блакитний = трава/кущі, жовтий = дерева/стовпи.
Беремо лише дорогу: рожевий -> 255, усе інше -> 0.
--name — як назвати «відео» для цих кадрів (за назвою ділимо на навчання/перевірку).
"""
import argparse
import glob
import os

import cv2
import numpy as np

ROAD_BGR = np.array([255, 0, 255])   # рожевий (magenta)


def find_root(p):
    for d, sub, _ in os.walk(p):
        if "frames" in sub and "markdown" in sub:
            return d
    raise SystemExit(f"У {p} не знайдено папок frames/ і markdown/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--data", default="data")
    ap.add_argument("--name", default="given")
    ap.add_argument("--tol", type=int, default=60, help="допуск кольору (маски можуть бути стиснуті)")
    a = ap.parse_args()

    root = find_root(a.src)
    os.makedirs(os.path.join(a.data, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.data, "masks"), exist_ok=True)
    n, frac = 0, []
    for mp in sorted(glob.glob(os.path.join(root, "markdown", "*.png"))):
        s = os.path.splitext(os.path.basename(mp))[0]
        ip = next((p for p in (os.path.join(root, "frames", s + e) for e in (".jpg", ".png", ".jpeg")) if os.path.exists(p)), None)
        if ip is None:
            continue
        m = cv2.imread(mp)
        img = cv2.imread(ip)
        road = (np.abs(m.astype(int) - ROAD_BGR).sum(2) < a.tol).astype(np.uint8) * 255
        if road.shape != img.shape[:2]:
            road = cv2.resize(road, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        out = f"{a.name}__{s}"
        cv2.imwrite(os.path.join(a.data, "images", out + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(a.data, "masks", out + ".png"), road)
        frac.append((road > 0).mean())
        n += 1
    print(f"Сконвертовано {n} кадрів у {a.data}/ (як відео «{a.name}»). "
          f"Дорога займає в середньому {np.mean(frac) * 100:.0f}% кадру.")


if __name__ == "__main__":
    main()
