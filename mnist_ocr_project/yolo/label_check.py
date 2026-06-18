import cv2, glob, os
imgs = sorted(glob.glob("dataset/images/train/*.jpg"))[::15]
os.makedirs("label_check", exist_ok=True)
for ip in imgs:
    lp = ip.replace("images","labels").replace(".jpg",".txt")
    im = cv2.imread(ip); H,W = im.shape[:2]
    with open(lp) as f:
        _,cx,cy,nw,nh = map(float, f.read().split())
    x1,y1 = int((cx-nw/2)*W), int((cy-nh/2)*H)
    x2,y2 = int((cx+nw/2)*W), int((cy+nh/2)*H)
    cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),5)
    cv2.imwrite(f"label_check/{os.path.basename(ip)}", cv2.resize(im,(300,533)))
print("저장 완료:", len(imgs), "장 -> label_check/")
