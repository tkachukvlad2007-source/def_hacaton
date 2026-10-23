#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Демо «круїз + виявлення дронів» на відео: модель дороги + YOLO-детектор дронів + супервізор guard.Guard.
Дрони (прямокутники / силуети) накладаються програмно в задані моменти.

  python guard_sim.py ../../data/dev_0008.mp4 --start 200 --frames 1600 \
      --drones 25:4:rect:a 55:6:rect:c 75:4:quad:a --operator-takes 80

  --drones T:DUR:ФОРМА:a|c   — дрон з'являється на T с (від --start), летить DUR с; a = наближається, c = пролітає
  --operator-takes T ...     — у ці моменти оператор бере керування (у вікні --show — клавіша M)
Пише results/<відео>_guard.mp4, _guard_events.txt (повідомлення оператору) і підсумок детекції.
"""
import argparse
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "road_seg"))
from collect_drive import yaw_shift  # noqa: E402
from drone_synth import Flight  # noqa: E402
from guard import Guard  # noqa: E402
from obstacles import Perception3, SynthObstacle, draw_classes, obstacle_ahead  # noqa: E402
from telemetry import Telemetry, video_yaw_lk  # noqa: E402
from seg_video import confidence, draw, keep_road, load_model, road_geometry, to_h264  # noqa: E402
from train_seg import preprocess  # noqa: E402

MODE_COL = {"MANUAL": (200, 200, 200), "CRUISE": (80, 220, 80), "THREAT": (60, 60, 255)}
LEVEL_COL = {"info": (120, 230, 120), "warn": (255, 200, 0), "alarm": (255, 80, 80)}   # RGB для PIL
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def hide_overlay(fr, boxes):
    """Замальовує вшитий текст рекордера (дата, лічильник кадрів) середнім тоном околу — щоб детектор
    не приймав цифри за дрон. boxes — частки кадру (x1, y1, x2, y2)."""
    h, w = fr.shape[:2]
    for bx in boxes:
        x1, y1, x2, y2 = int(bx[0] * w), int(bx[1] * h), int(bx[2] * w), int(bx[3] * h)
        ring = fr[max(0, y1 - 6):min(h, y2 + 6), max(0, x1 - 6):min(w, x2 + 6)]
        fr[y1:y2, x1:x2] = ring.reshape(-1, 3).mean(0).astype(np.uint8)
    return fr


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u else 0.0


class RouteMap:
    """Карта маршруту (вид згори) на чорному тлі: куди машина проїхала за командами системи,
    колір ділянки = режим, позначки дронів і моментів, коли оператор брав керування."""

    def __init__(self, fps, width=360, height=360, turn_rate=40.0, v_scale=1.0, fov=90.0):
        self.dt, self.w, self.h = 1.0 / fps, width, height
        self.turn, self.v_scale, self.fov = math.radians(turn_rate), v_scale, math.radians(fov)
        self.x = self.y = self.head = 0.0
        self.pts = [(0.0, 0.0, "MANUAL")]
        self.marks = []                           # (x, y, тип)
        self.back = None                          # індекс у pts, по якому відступаємо назад
        self.head_fix = 0.0                       # поправка курсу після розворотів (телеметрія не знає про них)

    def step(self, cmd, yaw, mode, action, threat_started, operator_took, real=None, move=1.0):
        """yaw — виміряний з відео поворот камери за кадр (частка ширини кадру, >0 праворуч).
        real — (швидкість м/с, курс рад) з телеметрії GPS/IMU, якщо є: тоді рух машини по карті —
        справжній (стоїмо — стоїмо), а не з команд. Поки система їде по дорозі, напрямок беремо
        з телеметрії або з відео; розворот і відступ у відео нема — тоді інтегруємо команди системи."""
        v = cmd["speed"] * self.v_scale * (move if real is None else 1.0)   # без телеметрії: стоїть картинка — стоїть карта
        if real is not None and action not in ("TURN", "RETREAT", "RESUME"):
            v = real[0]
            self.head = real[1] + self.head_fix
        if action == "TURN" and self.back is None:
            self.back = len(self.pts) - 1
            if real is not None:                 # далі курс інтегруємо від команд, потім повернемось до телеметрії
                self.head_fix = self.head - real[1]
        if action == "RETREAT" and self.back is not None:   # відступ = назад по власному сліду
            d = abs(v) * self.dt
            while d > 1e-6 and self.back > 0:
                x1, y1, _ = self.pts[self.back - 1]
                seg = math.hypot(x1 - self.x, y1 - self.y)
                if seg <= d:
                    d -= seg
                    self.x, self.y, self.back = x1, y1, self.back - 1
                else:
                    self.head = math.atan2(x1 - self.x, y1 - self.y)
                    self.x += d * math.sin(self.head)
                    self.y += d * math.cos(self.head)
                    d = 0.0
            if d > 1e-6:                                   # слід закінчився — далі прямо
                self.x += d * math.sin(self.head)
                self.y += d * math.cos(self.head)
        else:
            if action in ("TURN", "RESUME"):
                self.head += cmd["steer"] * self.turn * self.dt
                if real is not None:
                    self.head_fix = self.head - real[1]
            elif real is None:
                self.head += yaw * self.fov
                self.back = None
            else:
                self.back = None
            self.x += v * self.dt * math.sin(self.head)
            self.y += v * self.dt * math.cos(self.head)
        self.pts.append((self.x, self.y, mode))
        if threat_started:
            self.marks.append((self.x, self.y, "drone"))
        if operator_took:
            self.marks.append((self.x, self.y, "operator"))

    def render(self, font, t):
        img = np.zeros((self.h, self.w, 3), np.uint8)
        p = np.array([(x, y) for x, y, _ in self.pts])
        span = max(np.ptp(p[:, 0]), np.ptp(p[:, 1]), 8.0) * 1.15
        sc = (min(self.w, self.h) - 40) / span
        cx, cy = (p[:, 0].max() + p[:, 0].min()) / 2, (p[:, 1].max() + p[:, 1].min()) / 2

        def to_px(x, y):
            return int(self.w / 2 + (x - cx) * sc), int(self.h / 2 - (y - cy) * sc)

        # сітка 5 м
        step = 5.0
        for k in np.arange(math.floor((cx - span) / step) * step, cx + span, step):
            x0 = to_px(k, 0)[0]
            cv2.line(img, (x0, 0), (x0, self.h), (35, 35, 35), 1)
        for k in np.arange(math.floor((cy - span) / step) * step, cy + span, step):
            y0 = to_px(0, k)[1]
            cv2.line(img, (0, y0), (self.w, y0), (35, 35, 35), 1)
        for (x0, y0, m0), (x1, y1, m1) in zip(self.pts[:-1], self.pts[1:]):
            cv2.line(img, to_px(x0, y0), to_px(x1, y1), MODE_COL[m1], 3, cv2.LINE_AA)
        for x, y, kind in self.marks:
            px, py = to_px(x, y)
            if kind == "drone":
                cv2.drawMarker(img, (px, py), (0, 0, 255), cv2.MARKER_TRIANGLE_UP, 14, 2)
            else:
                cv2.circle(img, (px, py), 6, (255, 255, 255), 2)
        # машина і напрямок
        px, py = to_px(self.x, self.y)
        tip = (int(px + 14 * math.sin(self.head)), int(py - 14 * math.cos(self.head)))
        cv2.circle(img, (px, py), 5, (255, 255, 255), -1)
        cv2.arrowedLine(img, (px, py), tip, (255, 255, 255), 2, tipLength=0.5)
        cv2.circle(img, to_px(0, 0), 4, (120, 120, 120), -1)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        d.text((8, 6), "Маршрут (вид згори)", font=font, fill=(200, 200, 200))
        leg = [("оператор", MODE_COL["MANUAL"]), ("автопілот", MODE_COL["CRUISE"]), ("загроза", MODE_COL["THREAT"])]
        for k, (name, col) in enumerate(leg):
            y = self.h - 24 * (len(leg) - k) - 4
            d.rectangle([8, y + 3, 22, y + 15], fill=col[::-1])
            d.text((28, y), name, font=font, fill=(200, 200, 200))
        d.text((8, self.h - 24 * (len(leg) + 1) - 4), "▲ дрон   ○ оператор взяв керування", font=font,
               fill=(200, 200, 200))
        dist = sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0, _), (x1, y1, _) in zip(self.pts[:-1], self.pts[1:]))
        d.text((8, 26), f"сітка 5 м · {t:.1f} с · пройдено {dist:.0f} м", font=font, fill=(160, 160, 160))
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def render(fr, mask, g, conf, dets, cmd, guard, link_ok, t, font, real=None, cls=None, obs=None, synth=None):
    h, w = fr.shape[:2]
    out = draw(fr, mask, g, None, conf, 0.6)
    if cls is not None:
        out = draw_classes(out, cls, obs)
    if synth is not None:
        out = synth.draw(out, g)
    mode = cmd["mode"].split(":")[0]
    col = MODE_COL[mode]
    if mode == "THREAT" and int(t * 4) % 2 == 0:            # миготлива червона рамка
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
    for x1, y1, x2, y2, c in dets:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(out, f"drone {c:.2f}", (int(x1), max(12, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 255), 1, cv2.LINE_AA)
    # кермо і швидкість
    bw, by = int(w * 0.4), h - 22
    x0 = w // 2 - bw // 2
    cv2.rectangle(out, (x0, by - 7), (x0 + bw, by + 7), (0, 0, 0), -1)
    xs = int(w // 2 + np.clip(cmd["steer"], -1, 1) * bw / 2)
    cv2.rectangle(out, (min(w // 2, xs), by - 5), (max(w // 2, xs), by + 5), col, -1)
    cv2.line(out, (w // 2, by - 10), (w // 2, by + 10), (255, 255, 255), 1)
    sx, sh, sy = w - 26, int(h * 0.35), h - 40
    cv2.rectangle(out, (sx, sy - sh), (sx + 12, sy), (0, 0, 0), -1)
    v = cmd["speed"]
    cv2.rectangle(out, (sx, int(sy - sh * min(abs(v), 1))), (sx + 12, sy), col if v >= 0 else (255, 140, 60), -1)
    # текст (кирилиця — через PIL)
    img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    names = {"MANUAL": "ОПЕРАТОР", "CRUISE": "АВТОПІЛОТ", "THREAT": "ЗАГРОЗА"}
    acts = {"TURN": "розворот", "RETREAT": "відступ", "DASH": "ривок уперед", "RESUME": "назад на маршрут",
            "ALERT": "тривога оператору", "STOP_OBST": "стоп, перешкода", "SLOW_OBST": "перешкоди поруч, повільно"}
    a = cmd["mode"].split(":")[1] if ":" in cmd["mode"] else ""
    label = names[mode] + (f" · {acts[a]}" if a else "") + ("" if link_ok else " · НЕМА ЗВ'ЯЗКУ")
    tw = d.textlength(label, font=font)
    d.rectangle([w - tw - 20, 6, w - 6, 32], fill=(0, 0, 0))
    d.text((w - tw - 13, 9), label, font=font, fill=col[::-1])
    sp = f"кермо {cmd['steer']:+.2f}   команда швидкості {v:+.2f}"
    if real is not None:
        sp += f"   GPS: {real[0] * 3.6:.1f} км/год"
    d.text((x0 - 40, by - 26), sp, font=font, fill=(255, 255, 255))
    recent = [e for e in guard.events if t - e[0] < 8][-3:]
    for k, (te, lvl, txt) in enumerate(recent):
        y = h - 60 - 22 * (len(recent) - k)
        d.rectangle([6, y - 2, 12 + d.textlength(txt, font=font) + 50, y + 19], fill=(0, 0, 0))
        d.text((10, y), f"{te:5.1f}с  {txt}", font=font, fill=LEVEL_COL[lvl])
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser(description="Демо: круїз + виявлення дронів над маршрутом")
    ap.add_argument("video")
    ap.add_argument("--seg-model", default=os.path.join(HERE, "..", "road_seg", "models", "road_seg_weather.pt"))
    ap.add_argument("--det-model", default="/data/guard/runs/drone/weights/best.pt")
    ap.add_argument("--drones", nargs="*", default=[], help="T:DUR:rect|quad|wing:a|c")
    ap.add_argument("--operator-takes", nargs="*", type=float, default=[])
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--det-conf", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--telemetry", default=None, help="папка з gps_log.txt та imu_log.txt (реальна швидкість і курс)")
    ap.add_argument("--video-start", default=None, help="годинник у першому кадрі відео, YYYYMMDDhhmmss")
    ap.add_argument("--log-offset", type=float, default=None, help="час логу, що відповідає 0 с відео")
    ap.add_argument("--speed-unit", default="m/s", choices=["m/s", "km/h"])
    ap.add_argument("--heading", default="imu", choices=["gps", "imu", "video", "imu+video"],
                    help="звідки курс для карти: GPS COG, гіроскоп IMU, зсув картинки, або злиття IMU+відео")
    ap.add_argument("--react", action="store_true", help="протидіяти дрону (ривок / розворот), а не лише тривожити")
    ap.add_argument("--hide", nargs="*", default=[], help="вшитий текст у кадрі, частки: x1:y1:x2:y2 ...")
    ap.add_argument("--obstacles", nargs="*", default=[],
                    help="симульовані перешкоди на шляху: T[:x] — з'являється на T с (x — положення, 0.5 = центр дороги); потрібні --mc-models")
    ap.add_argument("--mc-models", nargs="*", default=None,
                    help="моделі train_mc.py (3 класи): дорога + рослинність + перешкоди; автопілот гальмує перед перешкодами")
    a = ap.parse_args()

    from ultralytics import YOLO
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    perc = Perception3(a.mc_models, dev) if a.mc_models else None
    seg, size = (None, None) if perc else load_model(a.seg_model, dev)
    det = YOLO(a.det_model)
    font = ImageFont.truetype(FONT, 15)

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 18.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.start * fps))
    w0, h0 = cap.get(3), cap.get(4)
    w, h = 640, int(640 * h0 / w0) // 2 * 2
    rng = np.random.default_rng(a.seed)
    flights = []
    for s in a.drones:
        t0, dur, kind, ap_ = s.split(":")
        flights.append(Flight(float(t0), float(dur), w, h, rng, kind, ap_ == "a", contrast=True))
    name = os.path.splitext(os.path.basename(a.video))[0]
    os.makedirs(a.out, exist_ok=True)
    out_path = os.path.join(a.out, f"{name}_guard.mp4")
    rmap = RouteMap(fps, width=h, height=h, v_scale=3.0)   # швидкість 1.0 = 3 м/с (~10 км/год)
    tel = Telemetry(a.telemetry, a.video_start, a.log_offset, a.speed_unit) if a.telemetry else None
    if tel is not None:
        print(f"телеметрія: 0 с відео = {tel.offset:.1f} с логу, курс з {a.heading}")
    wr = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w + rmap.w, h))

    guard = Guard(fps=fps, det_conf=a.det_conf, react=a.react)
    synths = []
    for s in a.obstacles:
        parts = s.split(":")
        synths.append(SynthObstacle(float(parts[0]), w, h, float(parts[1]) if len(parts) > 1 else 0.5))
    last_speed = 0.0
    hide = [tuple(float(v) for v in hb.split(":")) for hb in a.hide]
    takes = sorted(a.operator_takes)
    prob = prev_mask = conf_s = prev_g = prev_top = win = None
    stats = {"gt": 0, "hit": 0, "fp_frames": 0, "frames": 0}
    modes, calm_n = {}, 0
    alarm_t = {}
    i, t_start, key_take, rate_s = 0, time.time(), False, 0.0
    head_int, imu_prev, lk_prev, prev_gray = None, None, 0.0, None   # для режимів video / imu+video
    while True:
        ok, fr = cap.read()
        if not ok or (a.frames and i >= a.frames):
            break
        t = i / fps
        fr = cv2.resize(fr, (w, h))
        if hide:
            fr = hide_overlay(fr, hide)
        gts = [b for b in (f.draw(fr, t) for f in flights) if b is not None]   # малюємо «дрон»
        cls = obs = None
        if perc is not None:
            p, cls = perc(fr)                      # p — ймовірність дороги вже згладжена в часі
            mask = keep_road(cls == 1)
        else:
            with torch.no_grad():
                p = torch.sigmoid(seg(preprocess(fr, size)[None].to(dev)))[0, 0].float().cpu().numpy()
            prob = p if prob is None else 0.5 * prob + 0.5 * p
            mask = keep_road(cv2.resize(prob, (w, h)) > 0.5)
        g = road_geometry(mask)
        synth_now = None
        for so in synths:
            so.step(t, last_speed, 1.0 / fps, guard.operator_has_control and guard.mode == "MANUAL" and t > so.t0 + 1.0)
            if so.active and cls is not None:
                cls = so.inject(cls, g)
                synth_now = so
        if cls is not None:
            obs = obstacle_ahead(cls, mask, g, w, h)
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        c = confidence(cv2.resize(p, (w, h)), prev_mask, mask, gray)
        conf_s = c if conf_s is None else 0.8 * conf_s + 0.2 * c
        prev_mask = mask
        small = cv2.resize(gray, (176, 96)).astype(np.float32)
        motion = float(np.abs(small - prev_g).mean()) if prev_g is not None else 0.0
        prev_g = small
        top = np.float32(gray[: int(h * 0.45)])
        if win is None:
            win = cv2.createHanningWindow(top.shape[::-1], cv2.CV_32F)
        yaw, resp = yaw_shift(prev_top * win, top * win) if prev_top is not None else (0.0, 0.0)
        prev_top = top
        yaw = float(np.clip(yaw, -0.03, 0.03)) if resp > 0.05 else 0.0
        r = det.predict(fr, imgsz=640, conf=0.25, verbose=False, device=0 if dev == "cuda" else "cpu")[0]
        dets = [(*b.xyxy[0].tolist(), float(b.conf[0])) for b in r.boxes]

        take = key_take or any(abs(t - tt) < 0.5 / fps for tt in takes)
        key_take = False
        # втручання оператора з телеметрії: машина реально повертає різко, а автопілот такого не командував
        if tel is not None and tel.covers(a.start + t) and i > 0:
            src = "imu" if a.heading != "gps" else "gps"
            rate = math.degrees(tel.heading(a.start + t, src) - tel.heading(a.start + t - 1 / fps, src)) * fps
            rate_s = 0.8 * rate_s + 0.2 * rate
            expected = guard.drv.steer * 40.0
            if abs(rate_s) > 20 and abs(rate_s - expected) > 20 and not guard.operator_has_control:
                take = True
                guard.say("info", "Оператор керує: різкий поворот")
        cmd = guard.step(t, g, conf_s, dets, motion, True, take, w, h, mask, obs)
        last_speed = cmd["speed"]
        move = float(np.clip((guard.motion_s - 1.5) / 2.5, 0.0, 1.0))   # картинка не рухається -> машина стоїть

        # статистика детекції (поріг як у супервізора)
        good = [d for d in dets if d[4] >= a.det_conf and not guard._on_ground(d, mask, w, h)]
        stats["frames"] += 1
        stats["gt"] += len(gts)
        stats["hit"] += sum(any(iou(gb, d[:4]) > 0.3 for d in good) for gb in gts)
        stats["fp_frames"] += any(all(iou(gb, d[:4]) <= 0.3 for gb in gts) for d in good)
        for k, f in enumerate(flights):
            if k not in alarm_t and f.t0 <= t <= f.t0 + f.dur and cmd["mode"].startswith("THREAT"):
                alarm_t[k] = t - f.t0

        modes[cmd["mode"]] = modes.get(cmd["mode"], 0) + 1
        calm_n += cmd["calm"]
        took = any(e[0] == t and e[2].startswith("Оператор") for e in guard.events[-2:])
        mode_, _, action_ = cmd["mode"].partition(":")
        tv = a.start + t
        real = None
        if tel is not None and tel.covers(tv):
            if a.heading in ("gps", "imu"):
                head = tel.heading(tv, a.heading)
            else:                                   # курс інтегруємо самі: з картинки та/або гіроскопа
                imu_now = tel.heading(tv, "imu")
                if head_int is None:
                    head_int, prev_gray = imu_now, gray
                else:
                    v_lk, rel = video_yaw_lk(prev_gray, gray)   # відстеження точок, витримує великі зсуви
                    lk_prev = v_lk if rel > 0 else lk_prev * 0.9   # точок не знайшли — тримаємо попередній поворот
                    d_video = lk_prev * rmap.fov
                    d_imu = imu_now - imu_prev
                    head_int += d_video if a.heading == "video" else 0.5 * (d_imu + d_video)
                imu_prev, prev_gray = imu_now, gray
                head = head_int
            real = (tel.speed(tv), head)
        rmap.step(cmd, yaw, mode_, action_, guard.threat_started, took, real, move)
        vis = np.hstack([render(fr, mask, g, conf_s, good, cmd, guard, True, t, font, real, cls, obs, synth_now),
                         rmap.render(font, t)])
        wr.write(vis)
        if a.show:
            cv2.imshow("guard", vis)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            key_take = k == ord("m")
        i += 1
        if i % 300 == 0:
            print(f"  {i} кадрів, {i / (time.time() - t_start):.1f} кадр/с", flush=True)
    cap.release()
    wr.release()
    to_h264(out_path)
    with open(os.path.join(a.out, f"{name}_guard_events.txt"), "w") as f:
        for te, lvl, txt in guard.events:
            f.write(f"{te:6.1f} с  [{lvl}]  {txt}\n")
    print("\nПовідомлення оператору:")
    for te, lvl, txt in guard.events:
        print(f"  {te:6.1f} с  [{lvl:5s}] {txt}")
    print("\nРежими: " + ", ".join(f"{k} {v / max(i, 1) * 100:.0f}%" for k, v in sorted(modes.items()))
          + f" | спокійних кадрів {calm_n / max(i, 1) * 100:.0f}%")
    print(f"\nДетекція: знайдено {stats['hit']}/{stats['gt']} кадрів з дроном "
          f"({stats['hit'] / max(stats['gt'], 1) * 100:.0f}%), хибні спрацювання на {stats['fp_frames']} з "
          f"{stats['frames']} кадрів")
    for k, f in enumerate(flights):
        lat = alarm_t.get(k)
        print(f"  політ {k + 1} ({f.kind}, {'наближається' if f.approach else 'пролітає'}, з {f.t0:.0f} с): "
              + (f"тривога через {lat:.2f} с" if lat is not None else "тривоги НЕ було"))
    print("->", out_path)


if __name__ == "__main__":
    main()
