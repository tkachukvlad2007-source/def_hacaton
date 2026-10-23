#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Супервізор «круїз + виявлення дронів»: зменшує навантаження на оператора.

  MANUAL  — керує оператор. Якщо ділянка спокійна (дорога видна, поворот плавний, картинка змінюється
            мало) calm_s секунд поспіль — сам вмикає CRUISE і повідомляє оператора.
  CRUISE  — їде сам по дорозі (control.Driver). Складна ділянка -> просить оператора взяти керування,
            а поки той не взяв — повзе.
  THREAT  — у кадрі дрон: тривога оператору «візьміть керування», але система не чекає і сама протидіє:
              дрон наближається (рамка швидко росте або вже велика) -> TURN (розворот на місці 180°)
              + RETREAT (повним ходом по дорозі назад, звідки приїхали);
              дрон далеко / пролітає збоку -> DASH (максимальна швидкість уперед по дорозі).
            Протидія триває, доки оператор не візьме керування або дрон не зникне на clear_s секунд.
  Без зв'язку — логіка control.Driver (AUTO / SLOW / STOP / RETURN), загроза так само обробляється.
"""
import math
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "road_seg"))
from control import Driver  # noqa: E402


class Guard:
    def __init__(self, fps=18.0, calm_s=2.0, calm_conf=0.7, calm_steer=0.5, calm_motion=8.0,
                 det_conf=0.5, det_frames=5, clear_s=3.0, turn_s=4.5, approach_rate=1.6, close_area=0.02,
                 react=False):
        """react=False — при дроні лише повідомляємо оператора, їдемо далі; True — ще й протидіємо
        (ривок / розворот і відступ)."""
        self.fps, self.dt, self.react = fps, 1.0 / fps, react
        self.drv = Driver(fps=fps)
        self.calm_need = int(calm_s * fps)
        self.calm_conf, self.calm_steer, self.calm_motion = calm_conf, calm_steer, calm_motion
        # часові константи — у секундах, перераховані в кадри (відео бувають 5 і 18 кадр/с)
        self.det_conf, self.clear_n = det_conf, int(clear_s * fps)
        self.det_frames = max(3, int(round(det_frames / 18.0 * fps)))   # 5 з 10 кадрів при 18 fps, ≥3 завжди
        self.miss_max = max(2, int(round(0.22 * fps)))
        self.static_decay = 0.9 ** (18.0 / fps)
        self.static_thr = max(1.5, 0.28 * fps)
        self.turn_n, self.approach_rate, self.close_area = int(turn_s * fps), approach_rate, close_area
        self.mode, self.action = "MANUAL", ""
        self.calm_run = self.hard_run = 0
        self.adv_s = self.motion_s = 0.0      # згладжені |кермо| і рух картинки
        self.seen_run = self.miss_run = 0
        self.areas = deque(maxlen=int(fps))          # площа найбільшого дрона за ~1 с
        self.det_win = deque(maxlen=self.clear_n)
        self.static_map = {}                         # клітинка кадру -> (скільки кадрів поспіль тут рамка, розмір)
        self.track = None                            # (cx, cy, розмір) останньої прийнятої рамки
        self.seen_hist = deque(maxlen=max(4, int(round(0.55 * fps))))   # чи був дрон в останніх ~0.55 с    # чи був дрон у кожному з останніх clear_s секунд кадрів
        self.turn_left, self.turn_dir, self.replay = 0, 1.0, []
        self.last_drone = None
        self.acked, self.ack_t = False, -1e9   # оператор узяв керування при цьому дроні — не тривожити повторно
        self.threat_started, self.threat_t = False, -1e9
        self.warned_close = False
        self.operator_has_control = True
        self.manual_until = -1e9   # після втручання оператора автопілот не вмикається хоча б hold_s
        self.hold_s = 5.0
        self.events = []                              # (t, рівень, текст) — повідомлення оператору
        self.t = 0.0
        # перешкоди на шляху (obstacles.obstacle_ahead): 0 чисто, 1 поруч — повільно, 2 на шляху — стоп
        self.obs_level, self.obs_run, self.block_run, self.obs_clear = 0, 0, 0, 0

    def say(self, level, text):
        self.events.append((self.t, level, text))

    def _start_turn(self):
        self.action, self.turn_left = "TURN", self.turn_n
        self.turn_dir = -1.0 if self.last_drone[1] > 0.5 else 1.0   # розвертаємось від дрона

    def _static(self, box):
        """Статичний об'єкт (ліхтар на горизонті, стовп): рамка на тому ж місці (клітинка 32 px) і того ж
        розміру тримається ~1 с і довше. Дрон, що летить на камеру, стоїть на місці, але РОСТЕ — не статичний."""
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        size = max(box[2] - box[0], box[3] - box[1])
        cell = (int(cx // 32), int(cy // 32))
        score, mean = self.static_map.get(cell, (0.0, size))
        static = score > self.static_thr and 0.7 < size / max(mean, 1) < 1.4
        self.static_map[cell] = (score + 1, 0.8 * mean + 0.2 * size)
        return static

    @staticmethod
    def _on_ground(d, road_mask, w, h):
        """Маленька рамка, що лежить на самій дорозі, — камінь / пляма, а не дрон: дрон летить над землею.
        Великий об'єкт лишаємо — дрон, що вже близько, може перекривати дорогу."""
        if road_mask is None:
            return False
        x1, y1, x2, y2 = (int(v) for v in d[:4])
        if (x2 - x1) * (y2 - y1) > 0.01 * w * h or (y1 + y2) / 2 < 0.5 * h:   # великий або у верхній половині — не земля
            return False
        roi = road_mask[max(0, y1):y2 + 1, max(0, x1):x2 + 1]
        return roi.size > 0 and roi.mean() > 0.6

    def _drone(self, dets, w, h, road_mask=None):
        """Найбільший дрон з достатньою впевненістю -> (площа частка кадру, x центру частка) або None.
        Рамка має продовжувати трек (бути поруч із попередньою): хибні рамки стрибають по кадру,
        справжній дрон рухається неперервно."""
        self.static_map = {k: (v[0] * self.static_decay, v[1]) for k, v in self.static_map.items() if v[0] > 0.1}
        good = [d for d in dets if d[4] >= self.det_conf and not self._on_ground(d, road_mask, w, h)
                and not self._static(d)]
        if not good:
            return None
        x1, y1, x2, y2, _ = max(good, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
        cx, cy, size = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)
        if self.track is not None:
            px, py, ps = self.track
            if math.hypot(cx - px, cy - py) > max(40.0, 2.0 * max(size, ps)):   # інший об'єкт — трек заново
                self.seen_hist.clear()
                self.areas.clear()
        self.track = (cx, cy, size)
        return (x2 - x1) * (y2 - y1) / (w * h), cx / w

    def step(self, t, g, conf, dets, motion, link_ok, operator_takes=False, w=640, h=360, road_mask=None,
             obstacle=None):
        """Один кадр. dets — [(x1, y1, x2, y2, conf)], motion — середня зміна кадру (0..255),
        operator_takes — оператор узяв керування (в симуляції — за таймером або клавішею),
        road_mask — маска дороги (bool HxW) для фільтра «об'єкт на землі»,
        obstacle — результат obstacles.obstacle_ahead() (level 0/1/2) або None, якщо перешкод не розпізнаємо."""
        self.t = t
        drone = self._drone(dets, w, h, road_mask)
        self.det_win.append(drone is not None)
        gone = (len(self.det_win) == self.det_win.maxlen and sum(self.det_win) < min(3, self.det_frames)
                and t - self.threat_t >= self.clear_n * self.dt)       # поодинокі хибні — не рахуємо
        self.seen_hist.append(drone is not None)
        if drone:
            self.miss_run = 0
            self.areas.append(drone[0])
            self.last_drone = drone
        else:
            self.miss_run += 1
            if self.miss_run > self.miss_max:         # трек обірвався
                self.track = None
                self.areas.clear()
                self.seen_hist.clear()
        self.seen_run = sum(self.seen_hist)           # скільки з останніх кадрів бачили дрон

        # --- загроза ---
        self.threat_started = False
        if self.acked and (gone or t - self.ack_t > 8.0):
            self.acked = False
        grow = 1.0
        if len(self.areas) == self.areas.maxlen:          # ріст площі за останню секунду, медіани проти тремтіння
            a, q = list(self.areas), max(1, self.areas.maxlen // 4)
            grow = float(np.median(a[-q:])) / max(float(np.median(a[:q])), 1e-9)
        approaching = drone is not None and (grow > self.approach_rate or self.areas[-1] > self.close_area)
        if self.mode != "THREAT" and self.seen_run >= self.det_frames and not self.acked:
            self.mode, self.operator_has_control, self.threat_started, self.threat_t = "THREAT", False, True, t
            if not self.react:
                self.action = "ALERT"
                self.say("alarm", "ЗАГРОЗА: дрон у кадрі! Візьміть керування")
            elif approaching:
                self._start_turn()
                self.say("alarm", "ДРОН наближається! Візьміть керування. Розворот і відступ")
            else:
                self.action = "DASH"
                self.say("alarm", "ДРОН у кадрі! Візьміть керування. Прискорення вперед")
        if self.mode == "THREAT":
            if self.action == "ALERT" and approaching and not self.warned_close:
                self.warned_close = True
                self.say("alarm", "ЗАГРОЗА: дрон наближається!")
            elif self.action == "DASH" and approaching:            # ескалація: дрон росте в кадрі
                self._start_turn()
                self.say("alarm", "ДРОН наближається! Розворот і відступ")
            if operator_takes:
                self.mode, self.action, self.operator_has_control = "MANUAL", "", True
                self.calm_run, self.acked, self.ack_t, self.manual_until = 0, True, t, t + self.hold_s
                self.say("info", "Оператор узяв керування")
            elif gone:
                self.warned_close = False
                if self.action in ("TURN", "RETREAT"):   # ми розвернулись — тепер назад на маршрут
                    self.action, self.turn_left = "RESUME", self.turn_n
                    self.say("info", "Дрон зник. Загрозу знято, повертаюсь на маршрут")
                else:
                    self.action = ""
                    self.say("info", "Дрон зник з кадру. Загрозу знято, автопілот їде далі")
                self.mode, self.calm_run, self.hard_run = "CRUISE", 0, 0
                self.operator_has_control = False

        # --- круїз ---
        adv = self.drv.target(g, w, h) if g is not None else 1.0
        self.adv_s = 0.9 * self.adv_s + 0.1 * abs(adv)
        self.motion_s = 0.9 * self.motion_s + 0.1 * motion
        calm = (g is not None and conf >= self.calm_conf and self.adv_s < self.calm_steer
                and self.motion_s < self.calm_motion and link_ok)
        hard = g is None or conf < 0.55 or self.adv_s > 0.7
        self.calm_run = self.calm_run + 1 if calm else 0
        self.hard_run = self.hard_run + 1 if hard else 0
        if self.mode == "MANUAL" and self.calm_run >= self.calm_need and t >= self.manual_until:
            self.mode, self.operator_has_control = "CRUISE", False
            self.say("info", "Спокійна ділянка: автопілот увімкнено")
        elif self.mode == "CRUISE":
            if operator_takes:
                self.mode, self.operator_has_control = "MANUAL", True
                self.calm_run, self.manual_until = 0, t + self.hold_s
                self.say("info", "Оператор узяв керування")
            elif self.hard_run == int(self.fps):
                self.say("warn", "Складна ділянка: візьміть керування (автопілот сповільнюється)")

        # --- команда ---
        drive_link = self.operator_has_control and link_ok
        cmd = self.drv.step(g, conf, drive_link, w, h)
        if self.mode == "CRUISE" and self.action == "RESUME":        # розворот назад на маршрут
            self.turn_left -= 1
            cmd["steer"], cmd["speed"] = -self.turn_dir, 0.0
            self.drv.steer, self.drv.speed = cmd["steer"], cmd["speed"]
            if self.turn_left <= 0:
                self.action = ""
        elif self.mode == "CRUISE" and self.hard_run >= self.fps:   # складна ділянка — повземо, поки оператор не візьме
            cmd["speed"] = self.drv.speed = min(self.drv.speed, 0.3)
        if self.mode == "THREAT" and self.action == "ALERT":
            pass                                   # лише тривога — їдемо, як їхали
        elif self.mode == "THREAT":
            if self.action == "TURN":
                self.turn_left -= 1
                cmd["steer"], cmd["speed"] = self.turn_dir, 0.0       # розворот на місці (гусениці)
                if self.turn_left <= 0:
                    self.action = "RETREAT"
            elif self.action == "RETREAT":   # після розвороту дорога знову перед камерою — їдемо по ній
                cmd["steer"], cmd["speed"] = (adv if g is not None else 0.0), (1.0 if g is not None else 0.3)
            elif self.action == "DASH":
                cmd["speed"] = 1.0
            self.drv.steer, self.drv.speed = cmd["steer"], cmd["speed"]
        obs_action = ""
        if obstacle is not None:
            lvl = obstacle["level"]
            self.obs_run = self.obs_run + 1 if lvl >= 1 else 0
            self.block_run = self.block_run + 1 if lvl == 2 else 0
            self.obs_clear = self.obs_clear + 1 if lvl == 0 else 0
            if self.block_run >= max(3, int(0.3 * self.fps)) and self.obs_level < 2:
                self.obs_level = 2
                self.say("alarm", "СТОП: перешкода на шляху. Візьміть керування")
            elif self.obs_run >= int(0.5 * self.fps) and self.obs_level == 0:
                self.obs_level = 1
                self.say("warn", "Перешкоди біля дороги: сповільнення")
            if self.obs_level and self.obs_clear >= int(1.5 * self.fps):
                self.obs_level = 0
                self.say("info", "Шлях вільний" + (", автопілот їде далі" if self.mode == "CRUISE" else ""))
            if self.mode == "CRUISE" and not self.action:      # автопілот їде сам — реагуємо швидкістю
                if self.obs_level == 2:
                    cmd["speed"] = self.drv.speed = 0.0
                    obs_action = "STOP_OBST"
                elif self.obs_level == 1:
                    cmd["speed"] = self.drv.speed = min(self.drv.speed, 0.5)
                    obs_action = "SLOW_OBST"
        act = self.action or obs_action
        cmd["mode"] = self.mode + (f":{act}" if act else "")
        cmd["calm"] = calm
        cmd["obstacle"] = self.obs_level
        return cmd
