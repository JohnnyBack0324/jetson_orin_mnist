#!/usr/bin/env python3
# page 8 구조와 동일한 LeNet 학습 -> conv1.bin 등 8개 가중치 raw float32 덤프
# cuDNN은 CROSS_CORRELATION(=PyTorch Conv2d와 동일)이라 커널 뒤집기 불필요
import os, glob
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Dataset

DEV    = "cuda" if torch.cuda.is_available() else "cpu"
DST    = "../mnistCUDNN/data"   # ★ get_path가 data/를 붙이므로 실제 읽히는 위치
MYDIR  = "my_digits_pgm"        # ★ convert_my_digits.py 결과 (라벨_번호.pgm)
REPEAT = 30
EPOCHS = 10

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.ip1   = nn.Linear(800, 500)
        self.ip2   = nn.Linear(500, 10)
        self.lrn   = nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=1.0)
    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)               # conv1 -> pool
        x = F.max_pool2d(self.conv2(x), 2)               # conv2 -> pool
        x = x.flatten(1)                                 # 50*4*4 = 800 (NCHW)
        x = F.relu(self.ip1(x))                          # ip1 -> ReLU
        x = self.lrn(x.view(-1, 500, 1, 1)).flatten(1)   # LRN over 500ch
        return self.ip2(x)                               # ip2 (softmax는 loss에서)

def read_pgm(path):
    with open(path, 'rb') as f:
        assert f.readline().strip() == b'P5'
        line = f.readline()
        while line.startswith(b'#'): line = f.readline()
        w, h = map(int, line.split())
        int(f.readline())  # maxval
        data = np.frombuffer(f.read(w*h), np.uint8).reshape(h, w)
    return data

# 증강: 살짝 회전/이동/스케일 — 촬영 각도·위치 변화에 강건 (통째 암기 방지)
aug = T.RandomAffine(degrees=12, translate=(0.1, 0.1), scale=(0.85, 1.15))

class MyDigits(Dataset):
    def __init__(self, folder, repeat):
        self.items = []
        for p in glob.glob(os.path.join(folder, "*.pgm")):
            lab = int(os.path.basename(p).split("_")[0])   # "6_000.pgm" -> 6
            self.items.append((read_pgm(p), lab))
        if not self.items:
            raise RuntimeError(f"no pgm in {folder} — convert_my_digits.py 먼저 실행")
        self.items = self.items * repeat
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        arr, lab = self.items[i]
        x = torch.tensor(arr.astype(np.float32)/255.0)[None]  # [1,28,28], 0~1
        return aug(x), lab

# ★ readImage 정규화와 일치: /255, 흰글씨/검은배경, mean 미차감
tf = transforms.ToTensor()
mnist_full = datasets.MNIST("data", train=True, download=True, transform=tf)
my = MyDigits(MYDIR, REPEAT)
print(f"MNIST: {len(mnist_full)} + my(6/8 x{REPEAT}): {len(my)}")
train  = ConcatDataset([mnist_full, my])
loader = DataLoader(train, batch_size=128, shuffle=True)

net   = LeNet().to(DEV)
opt   = torch.optim.Adam(net.parameters(), 1e-3)
lossf = nn.CrossEntropyLoss()
for ep in range(EPOCHS):
    net.train(); tot = 0
    for x, y in loader:
        x, y = x.to(DEV), y.to(DEV)
        opt.zero_grad(); l = lossf(net(x), y); l.backward(); opt.step()
        tot += l.item()
    print(f"epoch {ep+1}: loss {tot/len(loader):.4f}")

def dump(t, path): t.detach().cpu().numpy().astype('<f4').tofile(path)
os.makedirs(DST, exist_ok=True)
for name in ["conv1", "conv2", "ip1", "ip2"]:
    L = getattr(net, name)
    dump(L.weight, os.path.join(DST, f"{name}.bin"))
    dump(L.bias,   os.path.join(DST, f"{name}.bias.bin"))
print("dumped .bin to", DST)