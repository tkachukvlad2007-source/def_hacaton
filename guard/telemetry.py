#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Телеметрія НРК з логів поруч із відео (gps_log.txt, imu_log.txt):

  [VTG] <t> speed=<v> COG=<курс, град>          — GPS, 10 Гц
  [ATT] <t> Roll=.. Pitch=.. Yaw=<рад>          — IMU, ~22 Гц
  Time synchronization: <YYYYMMDDhhmmss>=<t>    — який момент годинника відповідає часу <t> у лозі

Час у лозі — лічильник секунд, не збігається з часом відео. Зв'язок: відео починається в момент
годинника, який написано у кадрі (накладка дати), а рядок синхронізації прив'язує годинник до <t>.
Якщо рядка немає — зсув задають вручну (--log-offset) або шукають за рухом (sync_by_motion).

  tel = Telemetry("/data/telemetry/z2", video_start="20260605003740")   # або log_offset=194.6
  tel.speed(t_video), tel.heading(t_video)     # м/с і курс у радіанах (0 = північ, за годинниковою)
"""
import math
import os
import re
from datetime import datetime

import numpy as np


def _parse(path, tag, keys):
    ts, vals, sync = [], [], None
    rx = re.compile(r"\[" + tag + r"\] ([\d.]+) " + " ".join(k + r"=([\-\d.]+)" for k in keys))
    for line in open(path, errors="ignore"):
        m = rx.match(line)
        if m:
            ts.append(float(m.group(1)))
            vals.append([float(m.group(i + 2)) for i in range(len(keys))])
        elif line.startswith("Time synchronization:"):
            a, b = line.split(":", 1)[1].strip().split("=")
            sync = (datetime.strptime(a.strip(), "%Y%m%d%H%M%S"), float(b))
    return np.array(ts), np.array(vals), sync


class Telemetry:
    def __init__(self, folder, video_start=None, log_offset=None, speed_unit="m/s"):
        """video_start — годинник у першому кадрі відео ('YYYYMMDDhhmmss'); log_offset — час логу,
        що відповідає 0 с відео (має пріоритет). speed_unit: 'm/s' або 'km/h'."""
        self.t_gps, g, sync_g = _parse(os.path.join(folder, "gps_log.txt"), "VTG", ["speed", "COG"])
        self.t_imu, a, sync_i = _parse(os.path.join(folder, "imu_log.txt"), "ATT", ["Roll", "Pitch", "Yaw"])
        k = 1 / 3.6 if speed_unit == "km/h" else 1.0
        self.v = g[:, 0] * k
        self.cog = np.radians(g[:, 1])
        yaw = np.unwrap(a[:, 2])
        d = np.diff(yaw)
        d[np.abs(d) > np.radians(10)] = 0.0         # викиди гіроскопа (>10° за 44 мс = 230°/с — нереально для НРК)
        self.yaw = np.concatenate([[yaw[0]], yaw[0] + np.cumsum(d)])
        self.sync = sync_g or sync_i
        if log_offset is not None:
            self.offset = float(log_offset)
        elif video_start is not None and self.sync is not None:
            dt = (datetime.strptime(video_start, "%Y%m%d%H%M%S") - self.sync[0]).total_seconds()
            self.offset = self.sync[1] + dt
        else:
            self.offset = float(self.t_gps[0])     # без синхронізації — вважаємо, що лог і відео стартували разом
        # курс: GPS COG ненадійний, коли машина стоїть (<0.5 м/с) — тоді тримаємо попередній
        cog = self.cog.copy()
        ok = self.v > 0.5
        if ok.any():
            idx = np.where(ok, np.arange(len(cog)), 0)
            idx = np.maximum.accumulate(idx)
            cog = cog[idx]
        self.cog_u = np.unwrap(cog)

    def _t(self, t_video):
        return self.offset + t_video

    def speed(self, t_video):
        """м/с"""
        return float(np.interp(self._t(t_video), self.t_gps, self.v))

    def heading(self, t_video, source="gps"):
        """радіани, 0 = північ, зростає за годинниковою; source: 'gps' (COG) або 'imu' (yaw, дрейфує)"""
        if source == "imu":
            return -float(np.interp(self._t(t_video), self.t_imu, self.yaw))   # yaw IMU проти годинникової
        return float(np.interp(self._t(t_video), self.t_gps, self.cog_u))

    def covers(self, t_video):
        t = self._t(t_video)
        return self.t_gps[0] - 2 <= t <= self.t_gps[-1] + 2     # np.interp на краях тримає крайнє значення

    def path(self, t0, t1, step=0.2, source="gps"):
        """Трек (x на схід, y на північ, м) з інтегрування швидкості й курсу — для перевірки."""
        xs, ys, x, y = [], [], 0.0, 0.0
        for t in np.arange(t0, t1, step):
            h, v = self.heading(t, source), self.speed(t)
            x += v * step * math.sin(h)
            y += v * step * math.cos(h)
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)


def sync_by_motion(folder, video, fps_sample=5.0, max_shift=400.0):
    """Шукає зсув логу відносно відео, коли рядка синхронізації нема: порівнює повороти камери
    з відео (зсув верхньої частини кадру між кадрами) з гіроскопом (швидкість зміни yaw IMU)."""
    import cv2
    t_imu, a, _ = _parse(os.path.join(folder, "imu_log.txt"), "ATT", ["Roll", "Pitch", "Yaw"])
    yaw = np.unwrap(a[:, 2])
    grid = np.arange(t_imu[0], t_imu[-1], 1 / fps_sample)
    imu_rate = np.gradient(np.interp(grid, t_imu, yaw), 1 / fps_sample)
    cap = cv2.VideoCapture(video)
    fps, n = cap.get(5), int(cap.get(7))
    step = max(1, int(round(fps / fps_sample)))
    prev, win, vid_rate, vid_t = None, None, [], []
    for i in range(0, n, step):
        cap.set(1, i)
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (352, 198))
        top = np.float32(g[:90])
        if win is None:
            win = cv2.createHanningWindow(top.shape[::-1], cv2.CV_32F)
        if prev is not None:
            (dx, _), resp = cv2.phaseCorrelate(prev * win, top * win)
            vid_rate.append(-dx / 352 if resp > 0.05 else 0.0)
            vid_t.append(i / fps)
        prev = top
    vid_rate = np.array(vid_rate) * fps_sample          # частка ширини кадру за секунду
    vid_t = np.array(vid_t)
    best = (-1.0, 0.0)
    vid_dur = n / fps
    lo, hi = -max_shift, max_shift
    if t_imu[-1] - t_imu[0] >= vid_dur:                  # відео коротше за лог — воно має лежати всередині логу
        lo, hi = 0.0, (t_imu[-1] - t_imu[0]) - vid_dur
    for shift in np.arange(lo, hi + 0.2, 0.2):           # offset = t_imu[0] + shift  <->  t_video = 0
        ti = grid - (t_imu[0] + shift)                    # час відео, якому відповідає кожна точка IMU
        r = np.interp(vid_t, ti, imu_rate, left=np.nan, right=np.nan)
        ok = ~np.isnan(r)
        if ok.sum() < 200:
            continue
        c = np.corrcoef(vid_rate[ok], -r[ok])[0, 1]       # знак: IMU yaw проти годинникової
        if c > best[0]:
            best = (c, t_imu[0] + shift)
    return best[1], best[0]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--video", default=None, help="знайти зсув за рухом (без рядка синхронізації)")
    a = ap.parse_args()
    if a.video:
        off, c = sync_by_motion(a.folder, a.video)
        print(f"зсув логу: {off:.1f} с відповідає 0 с відео (кореляція поворотів {c:.2f})")


def video_yaw_lk(prev_gray, gray, state=None):
    """Поворот камери між двома кадрами з картинки: відстежуємо точки у верхній частині кадру
    (далекий фон, там зсув ≈ лише від повороту) пірамідальним методом Лукаса–Канаде, беремо медіану
    горизонтального зсуву. Повертає зсув у частках ширини кадру (>0 = поворот праворуч) і надійність 0..1."""
    import cv2
    h, w = gray.shape
    band = slice(0, int(h * 0.45))
    p0 = cv2.goodFeaturesToTrack(prev_gray[band], maxCorners=300, qualityLevel=0.01, minDistance=8)
    if p0 is None or len(p0) < 20:
        return 0.0, 0.0
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray[band], gray[band], p0, None, winSize=(21, 21), maxLevel=4,
                                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    ok = st.ravel() == 1
    if ok.sum() < 20:
        return 0.0, 0.0
    d = (p1 - p0)[ok, 0]
    dx = float(np.median(d[:, 0]))
    spread = float(np.median(np.abs(d[:, 0] - dx)))          # розкид: малий = точки рухаються узгоджено
    return dx / w, float(np.clip(1 - spread / 10.0, 0, 1))   # знак узгоджено з heading() на zip2 (див. README)
