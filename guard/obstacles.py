#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сприйняття на 3 класи (дорога / рослинність / перешкода) для автопілота + оцінка «перешкода на шляху».

Perception3 — ансамбль моделей train_mc.py (задача 2): дає ймовірність дороги (для старої логіки
керування й упевненості) і карту класів із тією ж постобробкою, що й у здачі (згладжування в часі,
обережний клас «перешкода», фільтр дрібних плям).

obstacle_ahead() — де перешкоди відносно шляху машини:
  level 0 — чисто;
  level 1 — перешкода біля дороги (стовп, паркан у смузі вздовж краю дороги в нижній половині кадру);
  level 2 — перешкода НА шляху (в коридорі навколо осьової лінії дороги в нижній половині кадру).
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "road_seg"))
from predict_mc import probs_for, remove_small_obstacles  # noqa: E402
from train_mc import load_mc_model  # noqa: E402


class Perception3:
    def __init__(self, paths, dev="cuda", smooth=0.5, obstacle_scale=0.6, min_obstacle_frac=0.004, keep_tall=0.3):
        self.models = []
        for p in paths:
            m, ck = load_mc_model(p, dev)
            self.models.append((m, tuple(ck["size"])))
        self.dev, self.smooth, self.scale, self.min_frac, self.keep_tall = dev, smooth, obstacle_scale, min_obstacle_frac, keep_tall
        self.prev = None

    def __call__(self, frame):
        """-> (ймовірність дороги HxW float, карта класів HxW uint8: 1 дорога, 2 рослинність, 3 перешкода)"""
        probs = probs_for(self.models, frame, self.dev, tta=False)
        if self.prev is not None and self.prev.shape == probs.shape:
            probs = (1 - self.smooth) * probs + self.smooth * self.prev
        self.prev = probs
        q = probs.copy()
        q[3] *= self.scale
        pred = (q[1:].argmax(0) + 1).astype(np.uint8)
        h, w = pred.shape
        pred = remove_small_obstacles(pred, q, int(self.min_frac * h * w), self.keep_tall)
        return probs[1], pred


def obstacle_ahead(cls, road_mask, g, w, h, near_band=0.10, block_frac=0.002, near_frac=0.003):
    """cls — карта класів, road_mask — маска дороги (одна область під машиною), g — road_geometry(mask).
    Повертає dict(level, block, near, blocking_mask, near_mask)."""
    ob = cls == 3
    out = dict(level=0, block=0.0, near=0.0, blocking_mask=None, near_mask=None)
    if not ob.any():
        return out
    lower = np.zeros((h, w), bool)
    lower[int(h * 0.5):] = True
    k = int(near_band * w) | 1
    band = cv2.dilate(road_mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    near = ob & band & lower & ~road_mask
    blocking = np.zeros((h, w), bool)
    if g is not None:
        rows, left, right = g["rows"], g["left"], g["right"]
        order = np.argsort(rows)
        ys = np.arange(int(h * 0.5), h)
        half = np.interp(ys, rows[order], ((right - left) / 4.0)[order])          # половина центральної половини дороги
        half = np.maximum(half, 0.05 * w)
        cx = np.polyval(g["coef"], ys)
        xs = np.arange(w)[None, :]
        corridor = np.abs(xs - cx[:, None]) < half[:, None]
        blocking[int(h * 0.5):] = corridor & ob[int(h * 0.5):]
    out["block"] = float(blocking.mean())
    out["near"] = float(near.mean())
    out["blocking_mask"], out["near_mask"] = blocking, near
    if out["block"] > block_frac:
        out["level"] = 2
    elif out["block"] > block_frac / 4 or out["near"] > near_frac:   # далека перешкода в коридорі — вже повільно
        out["level"] = 1
    return out


def draw_classes(out, cls, obs=None):
    """Накладає рослинність (ледь помітно) і перешкоди (жовтим; ті, що на шляху — червоним) на кадр HUD."""
    ov = out.copy()
    ov[cls == 2] = (200, 200, 60)
    ov[cls == 3] = (0, 220, 255)
    out = cv2.addWeighted(ov, 0.25, out, 0.75, 0)
    cnts, _ = cv2.findContours((cls == 3).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 220, 255), 2, cv2.LINE_AA)
    if obs is not None and obs["blocking_mask"] is not None and obs["level"] == 2:
        cnts, _ = cv2.findContours(obs["blocking_mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 0, 255), 3, cv2.LINE_AA)
    return out


class SynthObstacle:
    """Симульована перешкода на шляху (для демо логіки «стоп перед перешкодою»; реальної перешкоди на
    тестових відео нема). Малюється в кадрі з підписом «симуляція» і записується в карту класів як клас 3,
    тобто перевіряється вся логіка після сприйняття: коридор шляху -> сповільнення -> стоп -> оператор."""

    def __init__(self, t0, w, h, x_frac=0.5, approach=0.12):
        self.t0, self.w, self.h, self.x_frac, self.approach = t0, w, h, x_frac, approach
        self.d = 1.0            # 1 далеко … 0 під бампером
        self.gone, self.slide, self.active = False, 0.0, False

    def step(self, t, speed, dt, operator_drives):
        if t < self.t0 or self.gone:
            self.active = False
            return
        self.active = True
        if operator_drives:                     # оператор об'їжджає: перешкода зміщується вбік і зникає
            self.slide += 0.35 * dt
            if self.slide > 0.6:
                self.gone, self.active = True, False
        else:
            self.d = max(0.12, self.d - self.approach * max(speed, 0.0) * dt * 3.0)   # наближається лише коли їдемо

    def box(self, g=None):
        """(x1, y1, x2, y2) у пікселях кадру."""
        k = 1.0 - self.d
        yb = int(self.h * (0.52 + 0.43 * k))
        bw, bh = int(self.w * (0.06 + 0.22 * k)), int(self.h * (0.08 + 0.30 * k))
        cx = self.w * (self.x_frac + self.slide)
        if g is not None and self.slide == 0.0:  # тримаємось осьової лінії дороги
            cx = float(np.polyval(g["coef"], yb)) + (self.x_frac - 0.5) * self.w
        x1, x2 = int(cx - bw / 2), int(cx + bw / 2)
        return max(0, x1), max(0, yb - bh), min(self.w - 1, x2), min(self.h - 1, yb)

    def inject(self, cls, g=None):
        """Вписати перешкоду в карту класів (якщо активна)."""
        if not self.active:
            return cls
        x1, y1, x2, y2 = self.box(g)
        cls[y1:y2, x1:x2] = 3
        return cls

    def draw(self, out, g=None):
        if not self.active:
            return out
        x1, y1, x2, y2 = self.box(g)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
        for k in range(y1, y2, 8):
            cv2.line(out, (x1, k), (x2, min(y2, k + 8)), (0, 200, 255), 1)
        cv2.putText(out, "OBSTACLE (simulated)", (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 200, 255), 1, cv2.LINE_AA)
        return out
