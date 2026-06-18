#!/usr/bin/env python3
# my_digits/6,8/*.jpg -> 28x28 pgm (흰글씨/검은배경)
# 가장 큰 연결요소 -> INTER_AREA 축소 -> 축소본에 Otsu 재적용(사진별 회색편차 자동대응)
import os, glob, cv2, numpy as np

SRC = "my_digits"
OUT = "my_digits_pgm"
os.makedirs(OUT, exist_ok=True)

def to_mnist28(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (b > 127).mean() > 0.5:
        b = 255 - b
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b)
    if n <= 1: return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    b = np.where(lab == biggest, 255, 0).astype(np.uint8)
    ys, xs = np.where(b > 127)
    if len(xs) < 20: return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    crop = b[y0:y1+1, x0:x1+1]
    h, w = crop.shape
    s = 20.0 / max(h, w)
    nw, nh = max(1, int(round(w*s))), max(1, int(round(h*s)))
    small = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    # ★ 진단으로 검증된 고정 임계값 10 — >10비율이 전 사진 0.2 근처로 일정했음
    small = np.where(small > 10, 255, 0).astype(np.uint8)
    if small.max() > 0:
        _, small = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    canvas = np.zeros((28, 28), np.uint8)
    yy, xx = (28-nh)//2, (28-nw)//2
    canvas[yy:yy+nh, xx:xx+nw] = small
    return canvas

def save_pgm(img28, path):
    with open(path, "wb") as f:
        f.write(b"P5\n28 28\n255\n")
        f.write(img28.tobytes())

for f in glob.glob(os.path.join(OUT, "*.pgm")):
    os.remove(f)

total = 0
for label in ["6", "8"]:
    files = sorted(glob.glob(os.path.join(SRC, label, "*.jpg")) +
                   glob.glob(os.path.join(SRC, label, "*.png")) +
                   glob.glob(os.path.join(SRC, label, "*.jpeg")))
    n = 0
    for i, fp in enumerate(files):
        img = cv2.imread(fp)
        if img is None: continue
        out = to_mnist28(img)
        if out is None: continue
        save_pgm(out, os.path.join(OUT, f"{label}_{i:03d}.pgm"))
        n += 1
    print(f"label {label}: {n}/{len(files)} converted")
    total += n
print("total:", total)