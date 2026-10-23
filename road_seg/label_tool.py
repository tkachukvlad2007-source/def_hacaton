#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простий розмітник дороги. Обводиш дорогу багатокутником — зберігається маска.

  python label_tool.py --images data/images --masks data/masks

Керування:
  ЛКМ            — додати точку (або схопити й тягнути наявну точку)
  Ctrl + ЛКМ     — вставити точку в найближчу сторону контуру (підправити форму)
  ПКМ            — прибрати найближчу точку
  Enter / Пробіл — зберегти й перейти до наступного кадру
  C              — скопіювати контур з попереднього розміченого кадру
  R              — очистити контур
  E              — дороги на кадрі немає (зберегти порожню маску)
  S              — пропустити кадр без збереження
  B              — повернутися на кадр назад
  Q / Esc        — вийти (усе вже збережене лишається)

Як обводити: ведеш точки по краях дороги (де кінчається проїзна частина і
починається трава/кущі), нижні точки ставиш прямо на нижній край кадру.
Наступний кадр автоматично отримує контур з попереднього — просто підсунь точки.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np


# коди кириличних клавіш (укр. розкладка) -> латинські команди
UA_KEYS = {0x6cb: ord("r"), 0x6eb: ord("r"), 0x6a6: ord("s"), 0x6b6: ord("s"), 0x6d5: ord("e"), 0x6f5: ord("e"),
           0x6d3: ord("c"), 0x6f3: ord("c"), 0x6c9: ord("b"), 0x6e9: ord("b"), 0x6ca: ord("q"), 0x6ea: ord("q"),
           ord("к"): ord("r"), ord("і"): ord("s"), ord("у"): ord("e"), ord("с"): ord("c"), ord("и"): ord("b"), ord("й"): ord("q")}


def save_mask(path, shape, pts):
    m = np.zeros(shape[:2], np.uint8)
    if len(pts) >= 3:
        cv2.fillPoly(m, [np.round(np.array(pts)).astype(np.int32)], 255)
    cv2.imwrite(path, m)


def seg_dist(p, a, b):
    p, a, b = map(np.asarray, (p, a, b))
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def main():
    ap = argparse.ArgumentParser(description="Розмітка дороги")
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--masks", default="data/masks")
    ap.add_argument("--start", type=int, default=None, help="з якого кадру почати (номер)")
    a = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(a.images, "*")) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        raise SystemExit(f"У {a.images} немає картинок")
    os.makedirs(a.masks, exist_ok=True)
    poly_path = os.path.join(a.masks, "_polygons.json")
    polys = json.load(open(poly_path, encoding="utf-8")) if os.path.exists(poly_path) else {}
    pre_path = os.path.join(a.masks, "_prelabels.json")   # авто-контури від моделі (prelabel.py)
    prelabels = json.load(open(pre_path, encoding="utf-8")) if os.path.exists(pre_path) else {}
    if prelabels:
        print(f"Знайдено авто-контури для {len(prelabels)} кадрів — перевіряй і тисни Enter, якщо все добре.")

    def stem(f):
        return os.path.splitext(os.path.basename(f))[0]

    def mask_path(f):
        return os.path.join(a.masks, stem(f) + ".png")

    def done():
        return sum(os.path.exists(mask_path(f)) for f in files)

    if a.start is not None:
        idx = min(max(0, a.start), len(files) - 1)
    else:
        idx = next((i for i, f in enumerate(files) if not os.path.exists(mask_path(f))), len(files) - 1)

    print(__doc__)
    st = {"pts": [], "drag": -1, "scale": 1.0}

    def on_mouse(ev, x, y, flags, _):
        s = st["scale"]
        X, Y = x / s, y / s
        pts = st["pts"]
        near = -1
        if pts:
            d = [np.hypot(px - X, py - Y) for px, py in pts]
            j = int(np.argmin(d))
            if d[j] < 12 / s:
                near = j
        if ev == cv2.EVENT_LBUTTONDOWN:
            if near >= 0:
                st["drag"] = near
            elif (flags & cv2.EVENT_FLAG_CTRLKEY) and len(pts) >= 3:
                n = len(pts)
                k = int(np.argmin([seg_dist((X, Y), pts[i], pts[(i + 1) % n]) for i in range(n)]))
                pts.insert(k + 1, [X, Y])
                st["drag"] = k + 1
            else:
                pts.append([X, Y])
        elif ev == cv2.EVENT_MOUSEMOVE and st["drag"] >= 0:
            pts[st["drag"]] = [X, Y]
        elif ev == cv2.EVENT_LBUTTONUP:
            st["drag"] = -1
        elif ev == cv2.EVENT_RBUTTONDOWN and pts:
            d = [np.hypot(px - X, py - Y) for px, py in pts]
            pts.pop(int(np.argmin(d)))

    win = "road labeling"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    def prev_polygon(i):
        for j in range(i - 1, -1, -1):
            p = polys.get(stem(files[j]))
            if p:
                return [list(q) for q in p]
        return []

    while 0 <= idx < len(files):
        f = files[idx]
        img = cv2.imread(f)
        if img is None:
            idx += 1
            continue
        h, w = img.shape[:2]
        st["scale"] = min(1.0, 1400 / w, 850 / h) if w > 1400 or h > 850 else 1.0
        copied = auto = False
        if stem(f) in polys:
            st["pts"] = [list(q) for q in polys[stem(f)]]
        elif prelabels.get(stem(f)):
            st["pts"] = [list(q) for q in prelabels[stem(f)]]
            auto = True
        else:
            st["pts"] = prev_polygon(idx)
            copied = bool(st["pts"])
        action = None
        while action is None:
            s = st["scale"]
            vis = cv2.resize(img, (int(w * s), int(h * s))) if s != 1.0 else img.copy()
            pts = st["pts"]
            if len(pts) >= 3:
                ov = vis.copy()
                cv2.fillPoly(ov, [np.round(np.array(pts) * s).astype(np.int32)], (60, 200, 60))
                vis = cv2.addWeighted(ov, 0.35, vis, 0.65, 0)
            if len(pts) >= 2:
                cv2.polylines(vis, [np.round(np.array(pts) * s).astype(np.int32)], len(pts) >= 3, (0, 165, 255), 2, cv2.LINE_AA)
            for k, (px, py) in enumerate(pts):
                cv2.circle(vis, (int(px * s), int(py * s)), 5, (0, 0, 255) if k == st["drag"] else (255, 255, 255), -1, cv2.LINE_AA)
            status = ("labeled" if os.path.exists(mask_path(f)) else
                      "AUTO by model - check & adjust" if auto else
                      "COPIED from previous - check & adjust" if copied else "new")
            txt = [f"{idx + 1}/{len(files)}  done: {done()}  {stem(f)}  [{status}]",
                   "LMB add/drag | Ctrl+LMB insert | RMB delete | Enter save | C copy | R reset | E no road | S skip | B back | Q quit"]
            for k, t in enumerate(txt):
                cv2.putText(vis, t, (10, 22 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(vis, t, (10, 22 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, vis)
            kx = cv2.waitKeyEx(20)
            k = UA_KEYS.get(kx, kx & 0xFF if kx >= 0 else 255)
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                action = "quit"
            elif k in (13, 10, 32):
                if len(st["pts"]) >= 3:
                    action = "save"
                else:
                    print("Потрібно хоча б 3 точки (або E, якщо дороги на кадрі немає)")
            elif k in (ord("c"), ord("C")):
                st["pts"] = prev_polygon(idx)
                copied = True
            elif k in (ord("r"), ord("R")):
                st["pts"] = []
            elif k in (ord("e"), ord("E")):
                st["pts"] = []
                action = "empty"
            elif k in (ord("s"), ord("S")):
                action = "skip"
            elif k in (ord("b"), ord("B")):
                action = "back"
            elif k in (ord("q"), ord("Q"), 27):
                action = "quit"

        if action in ("save", "empty"):
            save_mask(mask_path(f), img.shape, st["pts"])
            polys[stem(f)] = st["pts"]
            with open(poly_path, "w", encoding="utf-8") as fh:
                json.dump(polys, fh)
            idx += 1
        elif action == "skip":
            idx += 1
        elif action == "back":
            idx = max(0, idx - 1)
        elif action == "quit":
            break
    cv2.destroyAllWindows()
    print(f"Розмічено кадрів: {done()} з {len(files)}. Маски в {a.masks}")


if __name__ == "__main__":
    main()
