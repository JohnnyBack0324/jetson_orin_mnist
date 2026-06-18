# !/usr/bin/env python3
# number.mp4 등에서 프레임 추출 + OpenCV 자동 박스 -> YOLO 1클래스 라벨 생성
# 검증된 detect 로직(area 기반) 재활용. 수작업 라벨링 불필요.
import os, glob, cv2, numpy as np

VIDEOS   = ["../mnistCUDNN/number.mp4"]   # 학습용 영상들 (추가 시 리스트에 append)
OUT_IMG  = "dataset/images/train"
OUT_LBL  = "dataset/labels/train"
EVERY    = 5            # 5프레임마다 1장 추출
MIN_AREA = 25000        # detect_opencv.py와 동일 기준
MAX_AREA = 200000

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

def detect_box(frame):
    """숫자 bounding box (x,y,w,h) 반환. 없으면 None."""
    g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (b > 127).mean() > 0.5:
        b = 255 - b
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b)
    if n <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = stats[biggest, cv2.CC_STAT_AREA]
    if area < MIN_AREA or area > MAX_AREA:
        return None
    x, y, w, h = (stats[biggest, cv2.CC_STAT_LEFT], stats[biggest, cv2.CC_STAT_TOP],
                  stats[biggest, cv2.CC_STAT_WIDTH], stats[biggest, cv2.CC_STAT_HEIGHT])
    return int(x), int(y), int(w), int(h)

idx = 0
for vpath in VIDEOS:
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        print("열 수 없음:", vpath); continue
    fno = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        fno += 1
        if fno % EVERY: continue
        box = detect_box(frame)
        if box is None: continue
        x, y, w, h = box
        H, W = frame.shape[:2]
        # YOLO 포맷: class cx cy w h (모두 0~1 정규화), class=0 (digit)
        cx, cy = (x + w/2) / W, (y + h/2) / H
        nw, nh = w / W, h / H
        cv2.imwrite(f"{OUT_IMG}/img_{idx:05d}.jpg", frame)
        with open(f"{OUT_LBL}/img_{idx:05d}.txt", "w") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        idx += 1
    cap.release()
    print(f"{vpath}: 누적 {idx}장")

print(f"\n총 {idx}장 라벨링 완료")