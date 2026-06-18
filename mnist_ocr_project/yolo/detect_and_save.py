#!/usr/bin/env python3
# 동영상 -> YOLO 검출 -> 등장당 정확히 1개 pgm 저장 (MNIST 28x28 규약)
import os, sys, cv2, numpy as np
from ultralytics import YOLO

VIDEO   = sys.argv[1] if len(sys.argv) > 1 else "number.mp4"
WEIGHTS = "best.pt"
OUTDIR  = "../pgm_output"

CONF       = 0.50   # 검출 신뢰도 임계값
GAP        = 8      # 이 (처리)프레임 수 이상 미검출이면 한 숫자 등장 종료
MIN_FRAME  = 3      # 너무 짧은 검출은 노이즈로 무시
FRAME_SKIP = 2      # N프레임마다 1번 추론 (녹화영상이라 무방)

os.makedirs(OUTDIR, exist_ok=True)
model = YOLO(WEIGHTS)

def sharpness(g):
    return cv2.Laplacian(g, cv2.CV_64F).var()

def to_mnist28(roi_bgr):
    g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(b > 0)
    if len(xs) == 0: return None
    b = b[ys.min():ys.max()+1, xs.min():xs.max()+1]   # 글자 타이트 크롭
    h, w = b.shape
    s = 20.0 / max(h, w)
    nw, nh = max(1, int(round(w*s))), max(1, int(round(h*s)))
    b = cv2.resize(b, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), np.uint8)             # 20x20을 28x28 중앙 배치
    y0, x0 = (28-nh)//2, (28-nw)//2
    canvas[y0:y0+nh, x0:x0+nw] = b
    return canvas

def save_pgm(img28, path):
    with open(path, "wb") as f:
        f.write(b"P5\n28 28\n255\n")
        f.write(img28.tobytes())

def best_box(frame):
    r = model(frame, conf=CONF, verbose=False)[0]
    if len(r.boxes) == 0: return None
    i = int(r.boxes.conf.argmax())
    x1, y1, x2, y2 = map(int, r.boxes.xyxy[i].tolist())
    return dict(conf=float(r.boxes.conf[i]), cls=int(r.boxes.cls[i]), box=(x1, y1, x2, y2))

saved = 0
def flush(seg):
    global saved
    cand = [s for s in seg if s[1] is not None]
    if not cand: return
    cand.sort(key=lambda s: s[0]*s[3], reverse=True)  # conf * 선명도
    conf, img28, cls, _, fnum = cand[0]
    saved += 1
    name = f"frame_{fnum:06d}_digit_{cls}_conf_{conf:.2f}_{saved:04d}.pgm"
    save_pgm(img28, os.path.join(OUTDIR, name))
    print("SAVE", name)

cap = cv2.VideoCapture(VIDEO)
seg = []; tracking = False; absence = 0; fno = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    fno += 1
    if fno % FRAME_SKIP != 0: continue
    det = best_box(frame)
    if det:
        absence = 0
        if tracking and seg and det["cls"] != seg[-1][2]:   # 클래스 바뀌면 마감
            flush(seg); seg = []
        tracking = True
        x1, y1, x2, y2 = det["box"]
        roi = frame[y1:y2, x1:x2]
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        seg.append((det["conf"], to_mnist28(roi), det["cls"], sharpness(g), fno))
    elif tracking:
        absence += 1
        if absence >= GAP:                              # 한 등장 종료
            if len(seg) >= MIN_FRAME: flush(seg)
            seg = []; tracking = False
if len(seg) >= MIN_FRAME: flush(seg)                    # 영상 끝 마감
cap.release()
print(f"\nTotal saved: {saved}")