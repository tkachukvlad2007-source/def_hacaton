#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Авто-розмітка (auto-labeling): модель + згладжування в часі + постобробка -> маски для відео або папки кадрів.

  python autolabel.py відео.mp4 --models M.pt --out /data/seg_mc/pseudo --every 30
  python autolabel.py папка_з_відео/ --models M1.pt M2.pt --tta --out /data/seg_mc/pseudo
  python autolabel.py кадри/frames --models M.pt --out розмітка --package

Що пише в --out:
  images/<відео>__<кадр>.jpg, masks/<відео>__<кадр>.png — наш формат для train_mc.py
      (0 небо, 1 дорога, 2 рослинність, 3 перешкода, 255 = модель не впевнена -> не вчимось на цьому пікселі)
  --package: ще й <відео>/frames/*.jpg + <відео>/markdown/*.png у кольорах організаторів
      (рожевий дорога, блакитний рослинність, жовтий перешкода, синій небо, чорний невпевнено) — пакет,
      який людина лише виправляє в будь-якому редакторі масок, і який читає їхній eval.py / наш prep_multiclass.py.
  --vis: відео з накладеною маскою для перегляду.

Невпевненість: піксель -> 255, якщо найбільша ймовірність < --conf (за замовчуванням 0.6) або
піксель на межі класів (ерозія на --edge px). Зберігається частка впевнених пікселів на кадр у conf.csv —
кадри з низькою часткою варто перевірити людині в першу чергу.
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np
import torch

from predict_mc import probs_for, remove_small_obstacles
from train_mc import load_mc_model

PALETTE_BGR = {0: (255, 0, 0), 1: (255, 0, 255), 2: (255, 255, 0), 3: (0, 255, 255)}
VIDEO_EXT = (".mp4", ".avi", ".mkv", ".mov")


def sources(path):
    """Список (назва, ітератор (номер кадру, кадр))."""
    if os.path.isdir(path):
        vids = sorted(p for p in glob.glob(os.path.join(path, "*")) if p.lower().endswith(VIDEO_EXT))
        if vids:
            for v in vids:
                yield os.path.splitext(os.path.basename(v))[0], video_frames(v)
        else:
            imgs = sorted(glob.glob(os.path.join(path, "*.jpg")) + glob.glob(os.path.join(path, "*.png")))
            name = os.path.basename(os.path.dirname(path.rstrip("/"))) or "frames"
            yield name, ((int(os.path.splitext(os.path.basename(p))[0]) if os.path.splitext(os.path.basename(p))[0].isdigit()
                          else i, cv2.imread(p)) for i, p in enumerate(imgs))
    else:
        yield os.path.splitext(os.path.basename(path))[0], video_frames(path)


def video_frames(path):
    cap = cv2.VideoCapture(path)
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield i, f
        i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="відео, папка з відео або папка з кадрами")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=30, help="зберігати кожен N-й кадр")
    ap.add_argument("--step", type=int, default=1, help="проганяти модель лише на кожному K-му кадрі (швидше; every має ділитись на K)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--smooth", type=float, default=0.5)
    ap.add_argument("--obstacle-scale", type=float, default=0.4)
    ap.add_argument("--min-obstacle", type=int, default=3000)
    ap.add_argument("--keep-tall", type=float, default=0.3)
    ap.add_argument("--conf", type=float, default=0.6)
    ap.add_argument("--edge", type=int, default=3, help="ерозія меж класів (px) -> невпевнено")
    ap.add_argument("--package", action="store_true")
    ap.add_argument("--vis", action="store_true")
    ap.add_argument("--max-width", type=int, default=0, help="зменшити кадр до цієї ширини перед збереженням (0 = ні)")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for p in a.models:
        m, ck = load_mc_model(p, dev)
        models.append((m, tuple(ck["size"])))
        print(f"{p}: {ck['arch']} {ck['encoder']} mIoU {ck.get('miou', float('nan')):.4f}")
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "masks"), exist_ok=True)
    csv_path = os.path.join(a.out, "conf.csv")
    new_csv = not os.path.exists(csv_path)
    cf = open(csv_path, "a", newline="")
    cw = csv.writer(cf)
    if new_csv:
        cw.writerow(["video", "frame", "confident_frac", "road_frac", "veg_frac", "obst_frac"])

    for name, frames in sources(a.src):
        prev, saved, writer = None, 0, None
        for i, img in frames:
            if i < a.start:
                continue
            if a.end is not None and i >= a.end:
                break
            if (i - a.start) % a.step:
                continue
            if img.std() < 6:   # чорний/порожній кадр (нема сигналу з камери) — пропускаємо
                continue
            if a.max_width and img.shape[1] > a.max_width:
                img = cv2.resize(img, (a.max_width, int(img.shape[0] * a.max_width / img.shape[1])), interpolation=cv2.INTER_AREA)
            probs = probs_for(models, img, dev, a.tta)
            if a.smooth > 0 and prev is not None and prev.shape == probs.shape:
                probs = (1 - a.smooth) * probs + a.smooth * prev
            prev = probs
            if (i - a.start) % a.every and not a.vis:
                continue
            q = probs.copy()
            q[3] *= a.obstacle_scale
            pred = q.argmax(0).astype(np.uint8)              # тут небо дозволене — це розмітка, не оцінка
            pred = remove_small_obstacles(pred, q, a.min_obstacle, a.keep_tall)
            conf = q.max(0) / q.sum(0)
            sure = conf >= a.conf
            if a.edge > 0:   # межі класів — невпевнено
                edges = cv2.Canny(pred * 60, 50, 150) > 0
                edges = cv2.dilate(edges.astype(np.uint8), np.ones((2 * a.edge + 1, 2 * a.edge + 1), np.uint8)) > 0
                sure &= ~edges
            lab = np.where(sure, pred, 255).astype(np.uint8)
            col = np.zeros_like(img)
            for k, c in PALETTE_BGR.items():
                col[lab == k] = c
            if a.vis:
                if writer is None:
                    os.makedirs(os.path.join(a.out, "vis"), exist_ok=True)
                    writer = cv2.VideoWriter(os.path.join(a.out, "vis", f"{name}_autolabel.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), 18, (img.shape[1], img.shape[0]))
                writer.write(cv2.addWeighted(img, 0.6, col, 0.4, 0))
                if (i - a.start) % a.every:
                    continue
            fid = f"{i:06d}"
            cv2.imwrite(os.path.join(a.out, "images", f"{name}__{fid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join(a.out, "masks", f"{name}__{fid}.png"), lab)
            if a.package:
                for sub in ("frames", "markdown"):
                    os.makedirs(os.path.join(a.out, name, sub), exist_ok=True)
                cv2.imwrite(os.path.join(a.out, name, "frames", fid + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(os.path.join(a.out, name, "markdown", fid + ".png"), col)
            tot = lab.size
            cw.writerow([name, i, f"{sure.mean():.3f}", f"{(lab == 1).sum() / tot:.3f}",
                         f"{(lab == 2).sum() / tot:.3f}", f"{(lab == 3).sum() / tot:.3f}"])
            saved += 1
        if writer is not None:
            writer.release()
        print(f"{name}: збережено {saved} кадрів", flush=True)
    cf.close()
    print(f"Готово -> {a.out} (conf.csv: частка впевнених пікселів на кадр)")


if __name__ == "__main__":
    main()
