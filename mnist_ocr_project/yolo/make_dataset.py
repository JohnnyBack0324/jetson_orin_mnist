#!/usr/bin/env python3
# 학습용 영상에서 프레임 추출 + 자동 bbox + 키 입력으로 클래스 라벨링
import os, cv2, glob

SRC_VIDEOS = "train_videos"          # 학습용 영상(.mp4)들을 여기 넣기
OUT_IMG = "dataset/images/train"
OUT_LBL = "dataset/labels/train"
EVERY = 10                           # 10프레임마다 1장

os.makedirs(OUT_IMG, exist_ok=True); os.makedirs(OUT_LBL, exist_ok=True)

def bbox_norm(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 500: return None
    x, y, w, h = cv2.boundingRect(c)
    H, W = frame.shape[:2]
    return (x+w/2)/W, (y+h/2)/H, w/W, h/H, (x, y, w, h)

idx = 0
for v in glob.glob(os.path.join(SRC_VIDEOS, "*.mp4")):
    cap = cv2.VideoCapture(v); fno = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        fno += 1
        if fno % EVERY: continue
        bb = bbox_norm(frame)
        if bb is None: continue
        cx, cy, bw, bh, (x, y, w, h) = bb
        disp = frame.copy()
        cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("label: 0-9 save / s skip / q quit", disp)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'): cap.release(); cv2.destroyAllWindows(); raise SystemExit
        if k == ord('s'): continue
        if ord('0') <= k <= ord('9'):
            cls = k - ord('0')
            cv2.imwrite(f"{OUT_IMG}/img_{idx:05d}.jpg", frame)
            with open(f"{OUT_LBL}/img_{idx:05d}.txt", "w") as f:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            idx += 1
    cap.release()
cv2.destroyAllWindows()
print("labeled:", idx)