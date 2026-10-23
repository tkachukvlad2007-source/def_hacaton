#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Синтетичні «дрони» на нічному відео: прямокутник (так ми симулюємо дрон), квадрокоптер, літак.
1) Датасет для YOLO з кадрів наших відео:
     python drone_synth.py dataset --videos ../../data/dev_0001.mp4 ... --val-videos dev_0008 --out /data/guard/ds
2) Політ дрона для демо — клас Flight (використовує guard_sim.py).
"""
import argparse
import os

import cv2
import numpy as np

KINDS = ("rect", "quad", "wing")


def shape_mask(kind, size, rng, angle=None):
    """Маска форми (float 0..1) розміром ~size px."""
    s = int(max(6, size))
    pad = s // 2 + 4
    m = np.zeros((s + 2 * pad, s + 2 * pad), np.float32)
    c = (m.shape[1] // 2, m.shape[0] // 2)
    if kind == "rect":
        ar = rng.uniform(1.0, 2.5)
        w, h = s, max(3, int(s / ar))
        cv2.rectangle(m, (c[0] - w // 2, c[1] - h // 2), (c[0] + w // 2, c[1] + h // 2), 1.0, -1)
    elif kind == "quad":
        r = s / 2
        th = max(1, int(s / 12))
        for a in (45, 135, 225, 315):
            e = (int(c[0] + r * 0.8 * np.cos(np.radians(a))), int(c[1] + r * 0.5 * np.sin(np.radians(a))))
            cv2.line(m, c, e, 1.0, th)
            cv2.ellipse(m, e, (max(2, int(r * 0.3)), max(1, int(r * 0.12))), 0, 0, 360, 1.0, -1)
        cv2.rectangle(m, (c[0] - int(r * 0.25), c[1] - int(r * 0.15)), (c[0] + int(r * 0.25), c[1] + int(r * 0.15)), 1.0, -1)
    else:  # wing
        r = s / 2
        th = max(2, int(s / 10))
        cv2.line(m, (int(c[0] - r), c[1]), (int(c[0] + r), c[1]), 1.0, th)
        cv2.line(m, (c[0], int(c[1] - r * 0.15)), (c[0], int(c[1] + r * 0.7)), 1.0, th)
        cv2.line(m, (int(c[0] - r * 0.3), int(c[1] + r * 0.65)), (int(c[0] + r * 0.3), int(c[1] + r * 0.65)), 1.0, th)
    a = rng.uniform(-25, 25) if angle is None else angle
    M = cv2.getRotationMatrix2D(c, a, 1.0)
    return cv2.warpAffine(m, M, m.shape[::-1], flags=cv2.INTER_LINEAR)


def paste(img, mask, cx, cy, value, alpha=1.0, blur=0.0, motion=(0.0, 0.0)):
    """Кладе форму в кадр (сірий uint8 або BGR). Повертає bbox (x1, y1, x2, y2) або None, якщо поза кадром."""
    if blur > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), blur)
    mx, my = motion
    k = int(max(abs(mx), abs(my)))
    if k >= 2:                      # змазування від руху дрона
        ker = np.zeros((2 * k + 1, 2 * k + 1), np.float32)
        cv2.line(ker, (k - int(mx / 2), k - int(my / 2)), (k + int(mx / 2), k + int(my / 2)), 1.0, 1)
        mask = cv2.filter2D(mask, -1, ker / max(ker.sum(), 1))
    H, W = img.shape[:2]
    h, w = mask.shape
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xb - xa < 3 or yb - ya < 3:
        return None
    m = mask[ya - y0:yb - y0, xa - x0:xb - x0] * alpha
    roi = img[ya:yb, xa:xb].astype(np.float32)
    mm = m[..., None] if roi.ndim == 3 else m
    img[ya:yb, xa:xb] = np.clip(roi * (1 - mm) + value * mm, 0, 255).astype(np.uint8)
    ys, xs = np.nonzero(m > 0.25)
    if len(xs) < 4:
        return None
    return xa + xs.min(), ya + ys.min(), xa + xs.max() + 1, ya + ys.max() + 1


def random_look(rng, bg):
    """Яскравість дрона: темний силует на світлому, світлий (ІЧ-підсвітка) або близький до фону."""
    r = rng.random()
    if r < 0.4:
        return float(rng.uniform(0, 60))
    if r < 0.8:
        return float(rng.uniform(170, 255))
    return float(np.clip(bg + rng.choice([-1, 1]) * rng.uniform(35, 80), 0, 255))


class Flight:
    """Політ одного дрона в кадрі: від точки появи по прямій, розмір росте (наближається) або ні."""

    def __init__(self, t0, dur, w, h, rng, kind="rect", approach=None, contrast=False):
        self.t0, self.dur, self.kind, self.contrast = t0, dur, kind, contrast
        self.approach = rng.random() < 0.5 if approach is None else approach
        contrast = self.contrast
        self.p0 = np.array([rng.uniform(0.1, 0.9) * w, rng.uniform(0.08, 0.45) * h])
        if self.approach:   # летить на нас: до центру-низу кадру, росте
            self.p1 = np.array([rng.uniform(0.35, 0.65) * w, rng.uniform(0.35, 0.6) * h])
            self.s0, self.s1 = rng.uniform(10, 18), rng.uniform(70, 120)
        else:               # пролітає збоку на відстані
            self.p1 = np.array([w - self.p0[0] + rng.uniform(-0.1, 0.1) * w, self.p0[1] + rng.uniform(-0.1, 0.1) * h])
            self.s0 = self.s1 = rng.uniform(22, 40) if contrast else rng.uniform(14, 30)
        self.value = None
        self.angle = rng.uniform(-15, 15)
        self.rng = rng

    def draw(self, img, t):
        """Малює дрон на момент t (с). Повертає bbox або None (дрона зараз нема)."""
        k = (t - self.t0) / self.dur
        if not 0 <= k <= 1:
            return None
        p = self.p0 + (self.p1 - self.p0) * k
        s = self.s0 * (self.s1 / self.s0) ** k        # експоненційний ріст = рівномірне наближення
        if self.value is None:
            g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            y0, x0 = int(np.clip(p[1], 0, g.shape[0] - 1)), int(np.clip(p[0], 0, g.shape[1] - 1))
            local = float(g[max(0, y0 - 30):y0 + 30, max(0, x0 - 30):x0 + 30].mean())
            # для демо — завжди контрастний до фону; для датасету — випадковий вигляд
            self.value = (25.0 if local > 110 else 235.0) if self.contrast else random_look(self.rng, float(g.mean()))
        v = (self.p1 - self.p0) / (self.dur * 18)      # px за кадр при 18 кадр/с (лише для змазування)
        return paste(img, shape_mask(self.kind, s, self.rng, self.angle + 4 * np.sin(t * 3)), p[0], p[1],
                     self.value, 0.95, 0.6, (v[0] * 3, v[1] * 3))


def make_dataset(a):
    rng = np.random.default_rng(0)
    val = set(a.val_videos)
    for sp in ("train", "val"):
        os.makedirs(os.path.join(a.out, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(a.out, "labels", sp), exist_ok=True)
    n_box = {"train": 0, "val": 0}
    for vp in a.videos:
        name = os.path.splitext(os.path.basename(vp))[0]
        sp = "val" if name in val else "train"
        cap = cv2.VideoCapture(vp)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        per = a.per_video_val if sp == "val" else a.per_video
        for j, fi in enumerate(np.sort(rng.choice(n, size=min(per, n), replace=False))):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, fr = cap.read()
            if not ok or fr.mean() < 15:         # чорні кадри (немає сигналу) пропускаємо
                continue
            fr = cv2.resize(fr, (640, int(640 * fr.shape[0] / fr.shape[1])))
            H, W = fr.shape[:2]
            lines = []
            k = rng.choice([0, 1, 1, 1, 2, 3])   # ~17% кадрів без дронів — вчимось не бачити їх у кущах
            for _ in range(k):
                s = float(np.exp(rng.uniform(np.log(8), np.log(130))))
                kind = KINDS[int(rng.choice(3, p=[0.5, 0.3, 0.2]))]
                cx, cy = rng.uniform(0.05, 0.95) * W, rng.uniform(0.03, 0.85) * H
                g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                val_ = random_look(rng, float(g[int(cy), int(cx)]))
                mot = (rng.normal(0, 4), rng.normal(0, 2)) if rng.random() < 0.4 else (0.0, 0.0)
                b = paste(fr, shape_mask(kind, s, rng), cx, cy, val_, rng.uniform(0.75, 1.0),
                          rng.uniform(0, 1.2), mot)
                if b:
                    x1, y1, x2, y2 = b
                    lines.append(f"0 {(x1 + x2) / 2 / W:.5f} {(y1 + y2) / 2 / H:.5f} {(x2 - x1) / W:.5f} {(y2 - y1) / H:.5f}")
            if rng.random() < 0.3:              # шум / стиснення, як у реальному стрімі
                fr = np.clip(fr + rng.normal(0, rng.uniform(2, 8), fr.shape), 0, 255).astype(np.uint8)
            stem = f"{name}_{j:04d}"
            q = int(rng.uniform(40, 95))
            cv2.imwrite(os.path.join(a.out, "images", sp, stem + ".jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, q])
            with open(os.path.join(a.out, "labels", sp, stem + ".txt"), "w") as f:
                f.write("\n".join(lines))
            n_box[sp] += len(lines)
        print(f"{name}: {sp}", flush=True)
    with open(os.path.join(a.out, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\nnames:\n  0: drone\n")
    print("дронів:", n_box, "->", os.path.join(a.out, "data.yaml"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dataset")
    d.add_argument("--videos", nargs="+", required=True)
    d.add_argument("--val-videos", nargs="*", default=["dev_0008"])
    d.add_argument("--out", default="/data/guard/ds")
    d.add_argument("--per-video", type=int, default=900)
    d.add_argument("--per-video-val", type=int, default=400)
    a = ap.parse_args()
    if a.cmd == "dataset":
        make_dataset(a)


if __name__ == "__main__":
    main()
