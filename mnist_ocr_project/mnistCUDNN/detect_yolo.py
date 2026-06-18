#!/usr/bin/env python3
# 영상 -> YOLO 검출 -> 등장당 1개 28x28 pgm 저장 (pgm_output/)
# 쿨다운 방식: 숫자 저장 후 일정시간 새 등장 차단 -> 영상 구성 무관하게 견고
import os, sys, cv2, numpy as np
from collections import deque
from ultralytics import YOLO

VIDEO   = sys.argv[1] if len(sys.argv) > 1 else "test.mp4"
WEIGHTS = "../yolo/best.pt"
OUTDIR  = "pgm_output"

CONF        = 0.30     # YOLO 검출 임계값
MIN_RATIO   = 0.10     # 박스 area 비율 이상이면 "숫자 있음"
FRAME_SKIP  = 3        # N프레임마다 처리
SMOOTH_N    = 1        # area_ratio 이동평균 (단발 튐 흡수)
ENTER_FRAMES = 3       # 이만큼 연속 검출되면 "등장 시작" 확정
EXIT_FRAMES  = 2       # 이만큼 연속 미검출이면 "등장 종료" 확정
COOLDOWN_SEC = 1.5     # ★ 한 숫자 저장 후 이 시간(초) 동안 새 등장 차단

model = YOLO(WEIGHTS)

def detect_ratio_mask(frame):
    r = model(frame, conf=CONF, verbose=False)[0]
    if len(r.boxes) == 0:
        return 0.0, None
    xyxy = r.boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2]-xyxy[:, 0])*(xyxy[:, 3]-xyxy[:, 1])
    i = int(np.argmax(areas))
    x1, y1, x2, y2 = map(int, xyxy[i])
    ratio = areas[i] / (frame.shape[0]*frame.shape[1])
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return ratio, None
    g = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5,5), 0)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (b > 127).mean() > 0.5:
        b = 255 - b
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        b = np.where(lab == biggest, 255, 0).astype(np.uint8)
    mask = np.zeros(frame.shape[:2], np.uint8)
    mask[y1:y2, x1:x2] = b
    return ratio, mask

def to_mnist28(mask):
    ys, xs = np.where(mask > 127)
    if len(xs) < 20:
        return None
    crop = mask[ys.min():ys.max()+1, xs.min():xs.max()+1]
    h, w = crop.shape
    s = 20.0/max(h, w)
    nw, nh = max(1, int(round(w*s))), max(1, int(round(h*s)))
    small = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    small = np.where(small > 10, 255, 0).astype(np.uint8)
    canvas = np.zeros((28,28), np.uint8)
    yy, xx = (28-nh)//2, (28-nw)//2
    canvas[yy:yy+nh, xx:xx+nw] = small
    return canvas

def save_pgm(img28, path):
    with open(path, "wb") as f:
        f.write(b"P5\n28 28\n255\n")
        f.write(img28.tobytes())

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    for f in os.listdir(OUTDIR):
        if f.endswith(".pgm"):
            os.remove(os.path.join(OUTDIR, f))

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print("영상 열기 실패:", VIDEO); return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cooldown_frames = int(COOLDOWN_SEC * fps)   # 쿨다운을 프레임 수로 환산

    seg = []                 # 현재 등장 구간 [(mask, fno, sharp), ...]
    present_run = 0          # 연속 검출 카운트
    absent_run  = 0          # 연속 미검출 카운트
    tracking = False
    cooldown_until = -1      # 이 프레임 번호 전까지는 새 등장 금지
    fno = 0
    saved = 0
    ratio_hist = deque(maxlen=SMOOTH_N)

    def flush():
        nonlocal saved, seg
        if not seg:
            return
        best = max(seg, key=lambda s: s[2])     # 선명도 best
        img = to_mnist28(best[0])
        if img is None:
            return
        saved += 1
        name = f"frame_{best[1]:06d}_digit_0_conf_0.99_{saved:04d}.pgm"
        save_pgm(img, os.path.join(OUTDIR, name))
        print("SAVE", name)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fno += 1
        if fno % FRAME_SKIP != 0:
            continue

        ratio, mask = detect_ratio_mask(frame)
        ratio_hist.append(ratio)
        smooth = sum(ratio_hist) / len(ratio_hist)
        present = (smooth >= MIN_RATIO and mask is not None)

        if present:
            present_run += 1
            absent_run = 0
        else:
            absent_run += 1
            present_run = 0

        if not tracking:
            # 쿨다운 중이 아니고, 충분히 연속 검출되면 등장 시작
            if fno >= cooldown_until and present_run >= ENTER_FRAMES:
                tracking = True
                seg = []
            # 추적 시작 직전 프레임도 담기 위해 present면 모아둠
            if present and fno >= cooldown_until:
                sharp = cv2.Laplacian(mask, cv2.CV_64F).var()
                seg.append((mask, fno, sharp))
        else:
            if present:
                sharp = cv2.Laplacian(mask, cv2.CV_64F).var()
                seg.append((mask, fno, sharp))
            # 충분히 연속 미검출이면 등장 종료 + 저장 + 쿨다운 시작
            if absent_run >= EXIT_FRAMES:
                flush()
                seg = []
                tracking = False
                cooldown_until = fno + cooldown_frames   # ★ 쿨다운 발동

    if tracking:
        flush()

    cap.release()
    print(f"\nTotal saved: {saved}")

if __name__ == "__main__":
    main()