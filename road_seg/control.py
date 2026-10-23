#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Керування наземним роботом з геометрії дороги + сценарій «втрата зв'язку». Без заліза: на вході — результат
road_geometry() і впевненість, на виході — команда (кермо -1..1, швидкість 0..1) і стан.

Стани:
  MANUAL  — зв'язок є, керує оператор; автопілот лише підказує команду
  AUTO    — зв'язку немає, їдемо самі по центру дороги
  SLOW    — дорогу видно погано (низька впевненість): повземо
  STOP    — дороги не видно / засвітка: стоїмо, чекаємо
  RETURN  — простояли довго без зв'язку: повертаємось назад, повторюючи записані команди у зворотному порядку
"""
from collections import deque

import numpy as np


class Driver:
    def __init__(self, fps=18.0, v_max=1.0, gain=1.0, full_lock=0.25, lookahead=0.7, lane=0.0,
                 conf_slow=0.6, conf_stop=0.35, stop_after=0.7, return_after=5.0, history_s=120):
        self.dt = 1.0 / fps
        self.v_max, self.gain, self.full_lock, self.lookahead, self.lane = v_max, gain, full_lock, lookahead, lane
        self.conf_slow, self.conf_stop = conf_slow, conf_stop
        self.stop_after, self.return_after = stop_after, return_after
        self.state, self.steer, self.speed = "MANUAL", 0.0, 0.0
        self.bad_t = 0.0          # скільки секунд поспіль дорогу видно погано
        self.stop_t = 0.0         # скільки секунд стоїмо без зв'язку
        self.history = deque(maxlen=int(history_s * fps))   # (кермо, швидкість) пройденого шляху
        self.replay = []

    def target(self, g, w, h):
        """Кермо з геометрії: наскільки точка центру дороги далеко попереду (lookahead — частка видимої
        дороги вгору) зсунута від центру кадру. >0 = праворуч. Повне кермо — коли зсув = full_lock ширини.
        lane — де триматись: 0 = по центру, +0.1 = на 10% ширини кадру правіше центру дороги.
        Так обрано за даними (collect_drive.py): на 5 відео водій повертав услід за далекою точкою
        приблизно через 1 с, а положення центру дороги біля машини поворот майже не передбачало."""
        yb, yt = g["rows"].max(), g["rows"].min()
        x_la = np.polyval(g["coef"], yb - self.lookahead * (yb - yt))
        err = (x_la - w / 2) / w + self.lane
        return float(np.clip(self.gain * err / self.full_lock, -1, 1))

    def step(self, g, conf, link_ok, w, h):
        """Один кадр. Повертає dict(state, steer, speed, advice) — advice = що порадив би автопілот."""
        ok_road = g is not None and conf >= self.conf_stop
        want = self.target(g, w, h) if g is not None else 0.0
        self.bad_t = 0.0 if (g is not None and conf >= self.conf_slow) else self.bad_t + self.dt

        if link_ok:
            if self.state == "RETURN":           # зв'язок повернувся під час руху назад — одразу стоп
                self.speed = 0.0
            self.state, self.stop_t, self.replay = "MANUAL", 0.0, []
        elif self.state == "RETURN":
            pass
        elif not ok_road and self.bad_t >= self.stop_after:
            self.state = "STOP"
        elif conf < self.conf_slow or g is None:
            self.state = "SLOW"
        else:
            self.state = "AUTO"

        if self.state == "STOP":
            self.stop_t += self.dt
            if self.stop_t >= self.return_after and self.history:
                self.state, self.replay = "RETURN", list(self.history)
        if self.state == "RETURN" and not self.replay:   # повернулись туди, де зв'язок ще був — стоїмо
            self.state = "STOP"
            self.history.clear()

        # швидкість: менше на поворотах і при гіршій впевненості
        v_auto = self.v_max * (1 - 0.5 * abs(want)) * float(np.clip((conf - self.conf_stop) /
                                                                     (0.8 - self.conf_stop), 0.2, 1))
        if self.state == "AUTO":
            steer_cmd, v_cmd = want, v_auto
        elif self.state == "SLOW":
            steer_cmd, v_cmd = (want if g is not None else self.steer), min(v_auto, 0.3 * self.v_max)
        elif self.state == "RETURN":
            s, v = self.replay.pop()            # остання команда першою; їдемо назад
            steer_cmd, v_cmd = s, -v
        else:                                    # MANUAL на відео — команд оператора нема, STOP — стоїмо
            steer_cmd, v_cmd = (want, v_auto) if self.state == "MANUAL" else (0.0, 0.0)

        # плавність: кермо не стрибає більше ніж на 2.0/с, швидкість — на 1.0/с
        self.steer += float(np.clip(steer_cmd - self.steer, -2.0 * self.dt, 2.0 * self.dt))
        if self.state == "RETURN":
            self.steer = steer_cmd               # повтор — точно ті самі команди
            self.speed = v_cmd
        else:
            self.speed += float(np.clip(v_cmd - self.speed, -1.0 * self.dt, 1.0 * self.dt))
        if self.state in ("AUTO", "SLOW", "MANUAL") and self.speed > 0:
            self.history.append((self.steer, self.speed))
        return dict(state=self.state, steer=self.steer, speed=self.speed, advice=want)
