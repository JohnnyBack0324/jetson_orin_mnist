# 손글씨 동영상 인식 파이프라인 (Jetson Orin + YOLO + cuDNN MNIST)

AI 가속기 기말 프로젝트 — 팀원: 백종윤, 최종윤 (숭실대학교 AI소프트웨어학부)

영상에 등장하는 손글씨 숫자를 **YOLO로 검출**하고, 등장당 1개씩 28×28 PGM으로 잘라낸 뒤,
**cuDNN으로 가속한 LeNet 분류기**로 숫자를 인식하는 엔드투엔드 파이프라인입니다.

```
영상(mp4) → YOLO 검출 → 디바운싱(등장당 1개) → 28×28 PGM(pgm_output/) → MNIST(cuDNN) 분류
```

---

## 1. 동작 환경

| 구분 | 내용 |
|------|------|
| 보드 | NVIDIA Jetson Orin Nano (aarch64, GPU sm_87) |
| OS / SDK | JetPack 6.x (Ubuntu 22.04 base, L4T 36.x) |
| GPU 가속 | CUDA 12.x, cuDNN 9.x, cuBLAS (JetPack 동봉) |
| 빌드 타깃 | `Makefile`의 `SMS := 80 86` + PTX forward-compat → sm_87(Orin)에서 JIT 실행 |

> x86_64 + NVIDIA GPU 데스크톱에서도 빌드/실행은 가능하지만, 본 프로젝트는 Jetson Orin 시연을 기준으로 작성되었습니다. 환경이 다르면 `mnistCUDNN/Makefile`의 `CUDA_PATH`·`SMS` 값을 채점/실행 환경에 맞게 조정하세요.

---

## 2. 의존성 (Dependencies)

의존성은 크게 **① C++ cuDNN 분류기(빌드)** 와 **② Python YOLO·학습 파이프라인** 으로 나뉩니다.

### 2-1. 시스템 패키지 (apt)

`mnistCUDNN` 빌드 및 시연 스크립트 실행에 필요합니다.

| 패키지 | 용도 |
|--------|------|
| `build-essential` | g++, make 등 빌드 도구 |
| CUDA Toolkit (`nvcc`) | `/usr/local/cuda` — cuDNN 분류기 컴파일/링크 |
| cuDNN / cuBLAS | `-lcudnn -lcublas -lcudart` 링크 (JetPack 동봉) |
| `libfreeimage-dev` | PGM/이미지 로딩 (`-lfreeimage`). **헤더만 동봉돼 있어 라이브러리는 시스템 설치 필요** |
| `bc` | `run_demo.sh`의 추론시간 합산 계산 |

```bash
sudo apt update
sudo apt install -y build-essential libfreeimage-dev bc
```

> JetPack을 정상 설치한 Jetson에는 CUDA·cuDNN·cuBLAS가 이미 `/usr/local/cuda`에 들어 있습니다.
> `nvcc --version` 으로 설치 여부를 확인하세요.

### 2-2. Python 패키지

| 패키지 | 용도 |
|--------|------|
| `python` 3.10 | 실행 인터프리터 |
| `pytorch` | YOLO 추론·LeNet 학습의 기반 (ultralytics 의존) |
| `torchvision` | MNIST 데이터셋·전처리 (학습 시) |
| `ultralytics` | YOLOv8 검출 (`detect_yolo.py`, 학습) |
| `opencv-python` | 프레임 처리·이진화·리사이즈 (`cv2`) |
| `numpy` | 배열 연산 |

> **⚠️ Jetson에서의 PyTorch 주의**
> Jetson(aarch64)에서는 `pip install torch`로 받는 일반 휠은 CUDA 가속이 되지 않습니다.
> 반드시 **JetPack 버전에 맞는 NVIDIA 공식 Jetson용 PyTorch 휠**을 설치해야 합니다.
> (NVIDIA Jetson PyTorch 포럼/공식 인덱스 참고. JetPack 6.x → `torch` for JetPack 6 휠)
> x86_64 GPU 환경에서는 PyTorch 공식 CUDA 빌드를 그대로 사용하면 됩니다.

---

## 3. Conda 가상환경 설정 (설치 방법)

### Step 1. 저장소 클론

```bash
git clone https://github.com/JohnnyBack0324/jetson_orin_mnist.git
cd jetson_orin_mnist/mnist_ocr_project
```

### Step 2. 시스템 의존성 설치

```bash
sudo apt update
sudo apt install -y build-essential libfreeimage-dev bc
nvcc --version    # CUDA(nvcc)가 보이는지 확인
```

### Step 3. Conda 가상환경 생성

Jetson에는 보통 Miniforge/Miniconda(aarch64)를 설치합니다. conda가 없다면 먼저 설치하세요.

```bash
# (conda 미설치 시) Miniforge aarch64 설치 예시
# wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
# bash Miniforge3-Linux-aarch64.sh

# 가상환경 생성 및 활성화
conda create -n mnist_ocr python=3.10 -y
conda activate mnist_ocr
```

### Step 4. PyTorch 설치 (환경에 맞게 택일)

**(A) Jetson Orin (aarch64, JetPack 6.x)** — NVIDIA 공식 Jetson 휠 사용:

```bash
# JetPack 버전에 맞는 NVIDIA 공식 torch 휠을 받아 설치
# (아래는 형식 예시 — 실제 파일명/URL은 본인 JetPack 버전에 맞춰 NVIDIA 인덱스에서 확인)
pip install --upgrade pip
pip install torch torchvision \
  --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126   # 예시: JetPack6 / CUDA 12.6
```

**(B) x86_64 + NVIDIA GPU 데스크톱** — PyTorch 공식 CUDA 빌드:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 5. 나머지 Python 패키지 설치

```bash
pip install ultralytics opencv-python numpy
```

> `ultralytics`가 의존성으로 torch를 끌어올 수 있으므로, **Step 4에서 올바른 torch를 먼저 설치**한 뒤 이 단계를 진행하세요. (잘못된 일반 torch로 덮어쓰이지 않게)

### Step 6. 설치 검증

```bash
python -c "import torch, cv2, ultralytics, numpy; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

`cuda True` 가 출력되면 GPU 가속 PyTorch가 정상입니다.

### Step 7. cuDNN 분류기 빌드 (최초 1회)

```bash
cd mnistCUDNN
make clean && make
```

빌드가 끝나면 `mnistCUDNN` 실행 파일이 생성됩니다. 단독 추론 테스트:

```bash
./mnistCUDNN image=data/three_28x28.pgm
# 출력 예:
#   Result of classification: 3
#   Inference time: 0.123 ms
```

> 채점/실행 환경의 GPU·CUDA 설정이 다르면, 동봉 `Makefile` 대신 환경에 맞는 Makefile로
> `mnistCUDNN.cpp`를 빌드하면 됩니다(소스 자체는 환경 독립적).

---

## 4. 실행 방법

### 4-1. (시연 전 1회) 보드 성능 최대화 — ★중요★

GPU 절전 상태로 시작하면 첫 추론이 8배 이상 느려집니다(예: 1697ms → 195ms).

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
```

### 4-2. 한 번에 실행 (검출 + 분류 + 채점)

`run_demo.sh` 상단 `answers=()` 배열에 영상 속 숫자를 **등장 순서대로** 입력한 뒤 실행:

```bash
cd mnistCUDNN
# 예) answers=(6 8 5)
./run_demo.sh <영상파일>      # 인자 생략 시 test.mp4 사용
```

→ YOLO 검출 → PGM 저장 → cuDNN 분류 → 정답 비교 → 정확도/추론시간 요약까지 한 번에 출력합니다.

### 4-3. 검출(PGM 생성)만 따로

```bash
cd mnistCUDNN
python3 detect_yolo.py <영상파일>      # pgm_output/ 에 등장당 1개 PGM 저장
ls pgm_output/*.pgm | wc -l            # 저장 개수 확인
```

파일명 형식: `frame_<프레임>_digit_0_conf_..._<등장순서>.pgm` (등장 순서가 시퀀스 번호로 보존됨)

### 4-4. 검출 개수가 안 맞을 때 (디바운싱 조정)

`detect_yolo.py` 상단 `COOLDOWN_SEC` 조정 후 다시 실행:

- 너무 많이 저장(한 숫자가 쪼개짐): 값 늘리기 (`1.5 → 2.0`)
- 너무 적게 저장(두 숫자가 합쳐짐): 값 줄이기 (`1.5 → 1.0`)

---

## 5. (선택) 모델 재학습

### YOLO 검출기 학습

```bash
cd yolo
python3 train_yolo.py
# 결과 -> runs/detect/digit_detector/weights/best.pt
```

### LeNet 분류기 학습 (.bin 가중치 생성)

```bash
cd mnist_train
python3 convert_my_digits.py   # 손글씨 사진 -> 28x28 pgm (my_digits_pgm/)
python3 train_lenet.py         # 학습 후 ../mnistCUDNN/data/ 에 .bin 8개 덤프
```

> cuDNN은 CROSS_CORRELATION(=PyTorch Conv2d와 동일)을 사용하므로 커널 뒤집기가 불필요합니다.

---

## 6. 폴더 구성

```
mnist_ocr_project/
├── mnistCUDNN/
│   ├── mnistCUDNN.cpp   # cuDNN 추론 + image 모드 Inference time 출력
│   ├── Makefile         # 빌드 설정 (CUDA_PATH=/usr/local/cuda, SMS=80 86)
│   ├── detect_yolo.py   # YOLO 검출 + 쿨다운 디바운싱 → pgm_output/ 저장
│   ├── run_demo.sh      # 검출+분류+채점 일괄 실행 (자체 리허설용)
│   ├── data/            # 학습된 가중치 .bin 8개 + 기본 테스트 pgm
│   ├── pgm_output/      # PGM 저장 위치 (실행 시 생성)
│   ├── FreeImage/       # FreeImage 헤더 (라이브러리는 시스템 설치 필요)
│   └── test.mp4         # 동작 확인용 (6, 8, 5)
├── yolo/
│   ├── best.pt          # 학습된 1클래스 digit 검출 모델
│   └── *.py             # YOLO 학습/라벨링 코드
└── mnist_train/
    ├── train_lenet.py       # LeNet 재학습 (.bin 생성)
    ├── convert_my_digits.py # 손글씨 사진 → pgm 변환
    └── my_digits_pgm/       # 변환된 학습용 pgm
```

---

## 7. mnistCUDNN 출력 형식 (채점 shell 호환)

```
Result of classification: 6      # 분류 결과
Inference time: 0.123 ms         # 순수 추론 시간
```

추론 시간 측정 구간: 변수 초기화 직후 ~ 추론 함수 시퀀스 ~ `cudaDeviceSynchronize()` 직후.