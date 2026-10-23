#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Прогін навченої моделі на відео: розмальовує дорогу, її краї й центральну лінію.

  python seg_video.py відео.mp4 --model models/road_seg.pt --out results
  python seg_video.py ПАПКА_З_ВІДЕО --model models/road_seg.pt --out results
  python seg_video.py відео.mp4 --model models/road_seg.pt --show      # живий перегляд у вікні
  python seg_video.py 0 --model models/road_seg.pt --show              # з камери (0 = перша камера)

На відео пишеться: ширина дороги, зсув від центру дороги і напрямок дороги попереду.
"""
import argparse
import glob
import math
import os
import shutil
import subprocess
import time

import cv2
import numpy as np
import torch

from train_seg import build_model, preprocess

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv")


def load_model(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = build_model(ck["encoder"], pretrained=False)
    m.load_state_dict(ck["state"])
    return m.to(dev).eval(), tuple(ck["size"])


def keep_road(mask):
    """Лишає одну область дороги — ту, що під машиною (біля низу кадру по центру)."""
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.zeros_like(m, bool)
    h, w = m.shape
    bottom = lab[int(h * 0.8):, int(w * 0.2):int(w * 0.8)]
    counts = np.bincount(bottom.ravel(), minlength=n)
    counts[0] = 0
    k = int(np.argmax(counts)) if counts.max() > 0 else int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
    return lab == k


def road_geometry(mask):
    """Краї й центр дороги по рядках знизу вгору."""
    h, w = mask.shape
    rows, left, right = [], [], []
    for y in np.linspace(h - 1, 0, 40).astype(int):
        xs = np.flatnonzero(mask[y])
        if len(xs) > 3:
            rows.append(y)
            left.append(xs[0])
            right.append(xs[-1])
    if len(rows) < 5:
        return None
    rows, left, right = map(np.asarray, (rows, left, right))
    center = (left + right) / 2
    deg = 2 if len(rows) >= 10 else 1
    c = np.polyfit(rows, center, deg)
    yb, yt = rows.max(), rows.min()
    xb, xt = np.polyval(c, yb), np.polyval(c, yt)
    width_b = right[0] - left[0]
    return dict(rows=rows, left=left, right=right, coef=c,
                width=width_b / w,                                    # ширина дороги внизу, частка кадру
                offset=(w / 2 - xb) / max(width_b, 1),                # >0: центр дороги лівіше, тобто машина правіше центру
                angle=math.degrees(math.atan2(xt - xb, max(yb - yt, 1))))  # >0: дорога йде праворуч


def glare_factor(gray):
    """Штраф за засвітку (фари, ліхтар): частка пересвічених пікселів у нижніх 2/3 кадру.
    На звичайних нічних відео її < 2% (рідко до ~15%), під фарами — 40%+. Модель при цьому може бути
    «впевнена», хоча дороги під плямою не видно, тому штраф рахуємо з самого кадру.
    До 10% — без штрафу, від 35% — впевненість 0."""
    sat = float(np.mean(gray[gray.shape[0] // 3:] >= 245))
    return float(np.clip(1 - (sat - 0.10) / 0.25, 0, 1))


def confidence(p, mask_prev, mask, gray=None):
    """Впевненість кадру 0..1: наскільки ймовірності близькі до 0/1 у зоні дороги та біля неї,
    помножено на стабільність маски відносно попереднього кадру і на штраф за засвітку."""
    zone = cv2.dilate(mask.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    if zone.sum() < 50:
        return 0.0
    certainty = float(np.mean(np.abs(2 * p[zone] - 1)))
    if gray is not None:
        certainty *= glare_factor(gray)
    if mask_prev is None:
        return certainty
    inter = np.logical_and(mask, mask_prev).sum()
    union = np.logical_or(mask, mask_prev).sum()
    stability = inter / union if union else 0.0
    return certainty * (0.5 + 0.5 * stability)


def draw(frame, mask, g, fps_val=None, conf=None, min_conf=0.0):
    low = conf is not None and conf < min_conf
    out = frame.copy()
    ov = out.copy()
    ov[mask] = (60, 60, 220) if low else (70, 200, 70)
    out = cv2.addWeighted(ov, 0.35, out, 0.65, 0)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    cv2.drawContours(out, cnts, -1, (0, 165, 255), max(2, w // 300), cv2.LINE_AA)
    lines = []
    if low:
        lines.append(f"LOW CONFIDENCE {conf:.2f} - SLOW DOWN")
    if g is not None and not low:
        ys = np.linspace(g["rows"].min(), g["rows"].max(), 30)
        cx = np.polyval(g["coef"], ys)
        cv2.polylines(out, [np.stack([cx, ys], 1).astype(np.int32)], False, (255, 255, 255), max(2, w // 400), cv2.LINE_AA)
        cv2.line(out, (w // 2, h - 1), (w // 2, int(h * 0.9)), (0, 0, 255), 2)
        a = g["angle"]
        d = "STRAIGHT" if abs(a) < 5 else ("RIGHT" if a > 0 else "LEFT")
        lines.append(f"road ahead: {d} {a:+.0f} deg")
        lines.append(f"width: {g['width'] * 100:.0f}%  offset: {g['offset']:+.2f}")
    elif g is None:
        lines.append("NO ROAD")
    if conf is not None and not low:
        lines.append(f"confidence: {conf:.2f}")
    if fps_val:
        lines.append(f"{fps_val:.0f} fps")
    fs = max(0.5, w / 1300)
    cv2.rectangle(out, (8, 8), (int(8 + 420 * fs * 1.3), int(14 + 32 * fs * len(lines))), (0, 0, 0), -1)
    for k, t in enumerate(lines):
        col = (60, 60, 255) if t.startswith("LOW") else (255, 255, 255)
        cv2.putText(out, t, (16, int(8 + 30 * fs * (k + 1))), cv2.FONT_HERSHEY_SIMPLEX, fs, col,
                    max(1, int(2 * fs)), cv2.LINE_AA)
    return out


def to_h264(path):
    if not shutil.which("ffmpeg"):
        return False
    tmp = path[:-4] + "_h264.mp4"
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-vcodec", "libx264",
                        "-pix_fmt", "yuv420p", tmp])
    if r.returncode == 0:
        os.replace(tmp, path)
        return True
    return False


def process(src, model, size, dev, a):
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        print(f"Не можу відкрити {src}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 <= fps <= 240 else 30.0
    w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = min(1.0, a.max_width / max(w0, 1))
    w, h = int(w0 * s) // 2 * 2, int(h0 * s) // 2 * 2
    wr, out_path = None, None
    if not a.no_save and not src.isdigit():
        os.makedirs(a.out, exist_ok=True)
        out_path = os.path.join(a.out, os.path.splitext(os.path.basename(src))[0] + "_road.mp4")
        wr = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    prob, i, t0, road_frames = None, 0, time.time(), 0
    fps_val, prev_mask, conf_s, low_frames, confs = None, None, None, 0, []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.resize(fr, (w, h))
        with torch.no_grad():
            p = torch.sigmoid(model(preprocess(fr, size)[None].to(dev)))[0, 0].float().cpu().numpy()
        prob = p if prob is None else a.smooth * prob + (1 - a.smooth) * p   # згладжування в часі
        mask = keep_road(cv2.resize(prob, (w, h)) > a.thr)
        g = road_geometry(mask)
        c = confidence(cv2.resize(p, (w, h)), prev_mask, mask,          # по сирому кадру, не згладженому
                       cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        conf_s = c if conf_s is None else 0.8 * conf_s + 0.2 * c
        prev_mask = mask
        confs.append(conf_s)
        low_frames += conf_s < a.min_conf
        road_frames += g is not None
        i += 1
        if i % 30 == 0:
            fps_val = i / (time.time() - t0)
        vis = draw(fr, mask, g, fps_val if a.show else None, conf_s, a.min_conf)
        if wr is not None:
            wr.write(vis)
        if a.show:
            cv2.imshow("road", vis)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
        if i % 300 == 0:
            print(f"  {os.path.basename(src)}: {i} кадрів, {i / (time.time() - t0):.1f} кадр/с", flush=True)
    cap.release()
    if wr is not None:
        wr.release()
        conv = to_h264(out_path)
        cf = np.array(confs) if confs else np.zeros(1)
        print(f"{src}: {i} кадрів, дорога знайдена на {road_frames / max(i, 1) * 100:.0f}%, "
              f"низька впевненість на {low_frames / max(i, 1) * 100:.0f}% кадрів "
              f"(впевненість: медіана {np.median(cf):.2f}, 10-й перцентиль {np.percentile(cf, 10):.2f}) -> {out_path}"
              + ("" if conv else "  (без ffmpeg відео може не грати в браузері)"))
    if a.show:
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Сегментація дороги на відео")
    ap.add_argument("inputs", nargs="+", help="відео, папки з відео або номер камери (0)")
    ap.add_argument("--model", default="models/road_seg.pt")
    ap.add_argument("--out", default="results")
    ap.add_argument("--show", action="store_true", help="показувати у вікні в реальному часі")
    ap.add_argument("--no-save", action="store_true", help="не записувати відео")
    ap.add_argument("--thr", type=float, default=0.5, help="поріг ймовірності дороги")
    ap.add_argument("--smooth", type=float, default=0.5, help="згладжування в часі, 0 = вимкнено")
    ap.add_argument("--max-width", type=int, default=960)
    ap.add_argument("--min-conf", type=float, default=0.6, help="нижче — LOW CONFIDENCE (підбирається по відео)")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, size = load_model(a.model, dev)
    print(f"Модель {a.model}, вхід {size[0]}x{size[1]}, пристрій {dev}")
    srcs = []
    for p in a.inputs:
        if os.path.isdir(p):
            srcs += [f for f in sorted(glob.glob(os.path.join(p, "*"))) if f.lower().endswith(VIDEO_EXT)]
        else:
            srcs.append(p)
    for s in srcs:
        process(s, model, size, dev, a)


if __name__ == "__main__":
    main()
