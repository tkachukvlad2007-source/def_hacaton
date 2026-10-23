#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Симуляція автопілота на відео: модель бачить дорогу -> control.Driver видає кермо й швидкість.
Зв'язок з оператором «зникає» за розкладом (--link-lost) або клавішею L у вікні (--show).

  python drive_sim.py відео.mp4 --model models/road_seg_weather.pt --link-lost 5-
  python drive_sim.py відео.mp4 --show --link-lost 10-40 60-       # L — перемкнути зв'язок вручну
  python drive_sim.py відео.mp4 --link-auto        # зв'язок «зник», коли відео зависло або почорніло

Пише results_drive/<відео>_drive.mp4 і <відео>_drive.csv (час, стан, впевненість, кермо, швидкість).
"""
import argparse
import csv
import math
import os
import time

import cv2
import numpy as np
import torch

from collect_drive import yaw_shift
from control import Driver
from seg_video import confidence, draw, keep_road, load_model, road_geometry, to_h264
from train_seg import preprocess

STATE_COL = {"MANUAL": (200, 200, 200), "AUTO": (80, 220, 80), "SLOW": (0, 200, 255),
             "STOP": (60, 60, 255), "RETURN": (255, 140, 60)}
# поворот водія (частка ширини кадру за кадр) -> одиниці керма; підібрано регресією в collect_drive.py
DRIVER_SCALE = 600.0


class LinkMonitor:
    """Стан зв'язку з самого відеопотоку (як його бачить оператор): кадр завис довше freeze_s або
    екран чорний -> LOST; багато завислих кадрів за останні 3 с -> WEAK (зв'язок слабне)."""

    def __init__(self, fps, freeze_s=0.5, weak_share=0.2):
        self.need, self.weak_share = int(freeze_s * fps), weak_share
        self.prev, self.run, self.recent = None, 0, []
        self.win = int(3 * fps)

    def update(self, frame):
        g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (176, 96)).astype(np.float32)
        frozen = self.prev is not None and float(np.abs(g - self.prev).mean()) < 0.05
        self.prev = g
        self.run = self.run + 1 if frozen else 0
        self.recent = (self.recent + [frozen])[-self.win:]
        if g.mean() < 15 or self.run >= self.need:
            return "LOST"
        return "WEAK" if np.mean(self.recent) > self.weak_share else "OK"


def parse_lost(specs):
    """'10-40' '60-' -> [(10, 40), (60, inf)] у секундах."""
    out = []
    for s in specs:
        a, _, b = s.partition("-")
        out.append((float(a or 0), float(b) if b else math.inf))
    return out


def hud(img, cmd, link, hist, conf):
    h, w = img.shape[:2]
    col = STATE_COL[cmd["state"]]
    # стан і зв'язок — праворуч угорі
    label = f"{cmd['state']}  link: {link}"
    fs = max(0.5, w / 1300)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
    cv2.rectangle(img, (w - tw - 24, 8), (w - 8, 20 + th), (0, 0, 0), -1)
    cv2.putText(img, label, (w - tw - 16, 14 + th), cv2.FONT_HERSHEY_SIMPLEX, fs, col, 2, cv2.LINE_AA)
    # кермо — шкала внизу по центру
    bw, by = int(w * 0.4), h - 30
    x0 = w // 2 - bw // 2
    cv2.rectangle(img, (x0, by - 8), (x0 + bw, by + 8), (0, 0, 0), -1)
    cv2.line(img, (w // 2, by - 12), (w // 2, by + 12), (255, 255, 255), 1)
    xs = int(w // 2 + cmd["steer"] * bw / 2)
    cv2.rectangle(img, (min(w // 2, xs), by - 6), (max(w // 2, xs), by + 6), col, -1)
    if cmd["state"] == "MANUAL":   # підказка автопілота
        xa = int(w // 2 + cmd["advice"] * bw / 2)
        cv2.circle(img, (xa, by), 6, (255, 255, 255), 2)
    cv2.putText(img, f"steer {cmd['steer']:+.2f}", (x0, by - 14), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8,
                (255, 255, 255), 1, cv2.LINE_AA)
    # швидкість — вертикальна смуга праворуч
    sx, sh = w - 30, int(h * 0.35)
    sy = h - 60
    cv2.rectangle(img, (sx, sy - sh), (sx + 14, sy), (0, 0, 0), -1)
    v = cmd["speed"]
    cv2.rectangle(img, (sx, int(sy - sh * min(abs(v), 1))), (sx + 14, sy), col if v >= 0 else (255, 140, 60), -1)
    cv2.putText(img, f"{v:+.2f}", (sx - 40, sy + 18), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, (255, 255, 255), 1,
                cv2.LINE_AA)
    # графік останніх 10 с: кермо автопілота (колір стану) і поворот водія з відео (білий)
    gw, gh = int(w * 0.3), int(h * 0.22)
    gx, gy = 10, h - gh - 60
    cv2.rectangle(img, (gx, gy), (gx + gw, gy + gh), (0, 0, 0), -1)
    cv2.line(img, (gx, gy + gh // 2), (gx + gw, gy + gh // 2), (90, 90, 90), 1)
    if len(hist) > 1:
        hst = np.array(hist)
        xs = gx + np.linspace(0, gw, len(hst))
        for k, colk in ((1, (255, 255, 255)), (0, col)):
            ys = gy + gh / 2 - np.clip(hst[:, k], -1, 1) * (gh / 2 - 4)
            cv2.polylines(img, [np.stack([xs, ys], 1).astype(np.int32)], False, colk, 1, cv2.LINE_AA)
    cv2.putText(img, "autopilot vs driver (white), 10 s", (gx + 4, gy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (200, 200, 200), 1)
    return img


def run(src, model, size, dev, a):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Не можу відкрити {src}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 <= fps <= 240 else 18.0
    w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = min(1.0, a.max_width / max(w0, 1))
    w, h = int(w0 * s) // 2 * 2, int(h0 * s) // 2 * 2
    if a.start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.start * fps))
    name = os.path.splitext(os.path.basename(src))[0]
    os.makedirs(a.out, exist_ok=True)
    out_path = os.path.join(a.out, name + "_drive.mp4")
    wr = None if a.no_save else cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    f_csv = open(os.path.join(a.out, name + "_drive.csv"), "w", newline="")
    log = csv.writer(f_csv)
    log.writerow(["t", "state", "link", "conf", "driver", "angle", "offset", "steer", "speed"])

    drv = Driver(fps=fps, conf_slow=a.min_conf, lane=a.lane)
    lost = parse_lost(a.link_lost)
    manual_toggle = False
    monitor = LinkMonitor(fps) if a.link_auto else None
    prob, prev_mask, conf_s = None, None, None
    hist, counts, i = [], {}, 0
    prev_top, win, yaws = None, None, []
    t_start = time.time()
    while True:
        ok, fr = cap.read()
        if not ok or (a.frames and i >= a.frames):
            break
        t = a.start + i / fps
        fr = cv2.resize(fr, (w, h))
        with torch.no_grad():
            p = torch.sigmoid(model(preprocess(fr, size)[None].to(dev)))[0, 0].float().cpu().numpy()
        prob = p if prob is None else a.smooth * prob + (1 - a.smooth) * p
        mask = keep_road(cv2.resize(prob, (w, h)) > 0.5)
        g = road_geometry(mask)
        c = confidence(cv2.resize(p, (w, h)), prev_mask, mask, cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        conf_s = c if conf_s is None else 0.8 * conf_s + 0.2 * c
        prev_mask = mask

        if monitor is not None:
            link = monitor.update(fr)
        else:
            link = "LOST" if any(lo <= t < hi for lo, hi in lost) else "OK"
        if manual_toggle:
            link = "LOST"
        link_ok = link != "LOST"
        cmd = drv.step(g, conf_s, link_ok, w, h)
        counts[cmd["state"]] = counts.get(cmd["state"], 0) + 1

        # як насправді повертав водій: зсув далекого фону (верх кадру) між кадрами, згладжено за 1 с
        top = np.float32(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)[: int(h * 0.45)])
        if win is None:
            win = cv2.createHanningWindow(top.shape[::-1], cv2.CV_32F)
        yw, resp = yaw_shift(prev_top * win, top * win) if prev_top is not None else (0.0, 0.0)
        prev_top = top
        yaws.append(float(np.clip(yw, -0.03, 0.03)) if resp > 0.05 else 0.0)
        driver = float(np.mean(yaws[-int(fps):])) * DRIVER_SCALE
        hist = (hist + [(cmd["steer"], driver)])[-int(10 * fps):]

        log.writerow([f"{t:.2f}", cmd["state"], link, f"{conf_s:.3f}", f"{driver:.3f}",
                      f"{g['angle']:.1f}" if g else "", f"{g['offset']:.3f}" if g else "",
                      f"{cmd['steer']:.3f}", f"{cmd['speed']:.3f}"])
        vis = hud(draw(fr, mask, g, None, conf_s, a.min_conf), cmd, link, hist, conf_s)
        if wr is not None:
            wr.write(vis)
        if a.show:
            cv2.imshow("drive", vis)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("l"):
                manual_toggle = not manual_toggle
        i += 1
        if i % 500 == 0:
            print(f"  {name}: {i} кадрів, {i / (time.time() - t_start):.1f} кадр/с", flush=True)
    cap.release()
    f_csv.close()
    if wr is not None:
        wr.release()
        to_h264(out_path)
    if a.show:
        cv2.destroyAllWindows()
    share = ", ".join(f"{k} {v / max(i, 1) * 100:.0f}%" for k, v in sorted(counts.items()))
    print(f"{name}: {i} кадрів | стани: {share} -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Симуляція автопілота НРК на відео")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--model", default="models/road_seg_weather.pt")
    ap.add_argument("--out", default="results_drive")
    ap.add_argument("--link-lost", nargs="*", default=["5-"], help="інтервали без зв'язку, сек: 10-40 60-")
    ap.add_argument("--link-auto", action="store_true",
                    help="визначати втрату зв'язку з самого відео (завислі / чорні кадри) замість --link-lost")
    ap.add_argument("--start", type=float, default=0.0, help="почати з секунди")
    ap.add_argument("--frames", type=int, default=0, help="обробити лише стільки кадрів (0 = все)")
    ap.add_argument("--min-conf", type=float, default=0.6)
    ap.add_argument("--lane", type=float, default=0.0, help="де триматись: +0.1 = правіше центру дороги")
    ap.add_argument("--smooth", type=float, default=0.5)
    ap.add_argument("--max-width", type=int, default=960)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, size = load_model(a.model, dev)
    for s in a.inputs:
        run(s, model, size, dev, a)


if __name__ == "__main__":
    main()
