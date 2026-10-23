#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Нарізка кадрів з відео для розмітки.

  python extract_frames.py ПАПКА_З_ВІДЕО --only dev_0001 dev_0002 --out data/images --every 1.5
  python extract_frames.py відео1.mp4 відео2.mp4 --out data/images

Кадри зберігаються як <назва_відео>__<номер_кадру>.jpg — за назвою потім ділимо
на навчання і перевірку саме по відео, а не по кадрах.
"""
import argparse
import glob
import os
import sys

import cv2

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv")


def find_videos(inputs, only):
    files = []
    for p in inputs:
        if os.path.isdir(p):
            files += [f for f in sorted(glob.glob(os.path.join(p, "*"))) if f.lower().endswith(VIDEO_EXT)]
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"Не знайдено: {p}")
    if only:
        want = {os.path.splitext(o)[0] for o in only}
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in want]
        found = {os.path.splitext(os.path.basename(f))[0] for f in files}
        for miss in sorted(want - found):
            print(f"УВАГА: відео {miss} не знайдено")
    return files


def main():
    ap = argparse.ArgumentParser(description="Нарізка кадрів з відео")
    ap.add_argument("inputs", nargs="+", help="відеофайли або папки з відео")
    ap.add_argument("--only", nargs="*", default=None, help="брати лише ці відео (назви без .mp4)")
    ap.add_argument("--out", default="data/images")
    ap.add_argument("--every", type=float, default=1.5, help="кадр кожні N секунд")
    ap.add_argument("--max-width", type=int, default=960)
    a = ap.parse_args()

    videos = find_videos(a.inputs, a.only)
    if not videos:
        sys.exit("Жодного відео не знайдено")
    os.makedirs(a.out, exist_ok=True)
    total = 0
    for v in videos:
        name = os.path.splitext(os.path.basename(v))[0]
        cap = cv2.VideoCapture(v)
        if not cap.isOpened():
            print(f"Не можу відкрити {v}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if 1 <= fps <= 240 else 30.0
        step = max(1, int(round(fps * a.every)))
        i = n = 0
        while True:
            if i % step == 0:
                ok, fr = cap.read()
                if not ok:
                    break
                h, w = fr.shape[:2]
                if w > a.max_width:
                    fr = cv2.resize(fr, (a.max_width, int(h * a.max_width / w) // 2 * 2), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(a.out, f"{name}__{i:06d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                n += 1
            elif not cap.grab():
                break
            i += 1
        cap.release()
        total += n
        print(f"{name}: {n} кадрів")
    print(f"Разом: {total} кадрів у {a.out}")


if __name__ == "__main__":
    main()
