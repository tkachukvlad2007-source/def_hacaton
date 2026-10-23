#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"Погана погода" для аугментації: засвітка фарами, дощ, краплі на лінзі, туман, темрява з шумом,
пересвітка. Працює з сірим кадром float 0..1 (HxW), маску не змінює — дорога лишається там, де була.

  from weather import add_weather
  f = add_weather(f, rng)          # випадковий ефект (або два)
"""
import cv2
import numpy as np


def glare(f, rng):
    """Зустрічні фари / ліхтар: 1-2 яскраві плями з насиченням, іноді горизонтальний відблиск."""
    h, w = f.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for _ in range(int(rng.integers(1, 3))):
        cx, cy = w * rng.uniform(0.1, 0.9), h * rng.uniform(0.2, 0.75)
        r = w * rng.uniform(0.08, 0.35)
        f = f + rng.uniform(0.6, 1.6) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / r ** 2)
        if rng.random() < 0.4:   # смуга відблиску лінзи
            f = f + rng.uniform(0.2, 0.6) * np.exp(-((yy - cy) / (h * 0.03)) ** 2 - ((xx - cx) / (w * 0.6)) ** 2)
    return np.clip(f, 0, 1)


def rain(f, rng):
    """Смуги дощу + розмиті краплі на лінзі."""
    h, w = f.shape
    f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.5, 1.5))
    streaks = np.zeros_like(f)
    dx = rng.uniform(-0.4, 0.4)
    for _ in range(int(rng.integers(100, 400))):
        x, y, ln = rng.integers(0, w), rng.integers(0, h), rng.integers(8, 35)
        cv2.line(streaks, (int(x), int(y)), (int(x + ln * dx), int(y + ln)), float(rng.uniform(0.4, 1.0)), 1)
    f = np.maximum(f, cv2.GaussianBlur(streaks, (3, 3), 0) * rng.uniform(0.4, 0.9))
    if rng.random() < 0.7:
        drops = np.zeros_like(f)
        for _ in range(int(rng.integers(3, 15))):
            cv2.circle(drops, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(5, 30)),
                       float(rng.uniform(0.5, 0.9)), -1)
        drops = cv2.GaussianBlur(drops, (0, 0), 4)
        f = f * (1 - drops) + drops * rng.uniform(0.5, 0.9)
    return np.clip(f, 0, 1)


def fog(f, rng):
    """Туман, густіший угорі кадру (далі від камери)."""
    h, _ = f.shape
    top, bot = rng.uniform(0.5, 0.9), rng.uniform(0.1, 0.5)
    k = np.linspace(top, bot, h, dtype=np.float32)[:, None]
    return f * (1 - k) + rng.uniform(0.3, 0.6) * k


def dark_noise(f, rng):
    """Слабка ІЧ-підсвітка: темний кадр, сильний шум сенсора."""
    f = f * rng.uniform(0.2, 0.5)
    return np.clip(f + rng.normal(0, rng.uniform(0.03, 0.09), f.shape).astype(np.float32), 0, 1)


def overexpose(f, rng):
    """Пересвічений кадр (автоекспозиція не встигла)."""
    return np.clip(f * rng.uniform(1.5, 3.0) + rng.uniform(0, 0.2), 0, 1)


EFFECTS = {"glare": glare, "rain": rain, "fog": fog, "dark_noise": dark_noise, "overexpose": overexpose}


def add_weather(f, rng, names=None):
    """Один випадковий ефект, з імовірністю 0.25 — два одразу (напр. дощ + фари)."""
    names = list(names or EFFECTS)
    k = 2 if rng.random() < 0.25 and len(names) > 1 else 1
    for n in rng.choice(names, size=k, replace=False):
        f = EFFECTS[n](f.astype(np.float32), rng)
    return f
