# AI 가속기 기말 프로젝트 — 손글씨 동영상 인식 파이프라인

팀원: 백종윤, 최종윤 (AI소프트웨어학부)

영상 → YOLO 검출 → 디바운싱(등장당 1개) → 28×28 PGM(pgm_output/) → MNIST(cuDNN) 분류

---

## 시연 실행 방법

### 1) (최초 1회) mnistCUDNN 빌드
```bash
cd mnist_ocr_project/mnistCUDNN
make clean && make
```
- 빌드는 채점 환경의 Makefile/그래픽카드 설정을 따릅니다(CUDA_PATH = /usr/local/cuda).
- 동봉된 Makefile은 시연 보드(Jetson Orin, sm_87)용 설정입니다. 채점 환경이 다르면
  채점자 Makefile을 사용해 mnistCUDNN.cpp 를 빌드하시면 됩니다(소스는 환경 독립적).

### 2) 영상 → PGM 생성 (pgm_output/ 에 저장)
```bash
cd mnist_ocr_project/mnistCUDNN
python3 detect_yolo.py <영상파일>
```
- 영상에 등장하는 손글씨를 등장당 1개씩 검출해 pgm_output/ 에 저장합니다.
- 파일명: frame_<프레임>_digit_0_conf_..._<등장순서>.pgm  (등장 순서가 시퀀스 번호로 보존됨)

### 3) 채점
- 채점 shell이 pgm_output/ 의 PGM을 등장 순서(sort -V)대로 앞 N개 읽어
  `./mnistCUDNN image=<pgm>` 로 분류하고, 출력의
  `Result of classification: X` (정답)과 `Inference time: X ms` (추론시간)를 파싱합니다.
- 동봉한 run_demo.sh 로 동일 방식의 자체 리허설이 가능합니다:
  ```bash
  ./run_demo.sh <영상파일>      # 상단 answers=() 배열에 정답을 등장 순서대로 입력
  ```

### mnistCUDNN 출력 형식 (채점 shell 호환)
```
Result of classification: 6
Inference time: 0.123 ms
```
- 추론 시간은 classify_example 내부에서 측정합니다:
  모든 변수 초기화 직후 ~ 추론 함수 시퀀스 ~ cudaDeviceSynchronize() 직후.

---

## 디바운싱 파라미터 (개수가 안 맞을 때)
detect_yolo.py 상단에서 조정:
- 너무 많이 저장(쪼개짐): COOLDOWN_SEC 늘리기 (1.5 → 2.0)
- 너무 적게 저장(합쳐짐): COOLDOWN_SEC 줄이기 (1.5 → 1.0)

---

## 폴더 구성
```
mnist_ocr_project/
├── mnistCUDNN/
│   ├── mnistCUDNN.cpp   # 추론 시간 측정(image 모드 Inference time 출력) + algo 고정 + 출력 제거
│   ├── Makefile         # 시연 보드(Jetson, sm_87)용 — 채점 환경에선 채점자 Makefile 사용 가능
│   ├── detect_yolo.py   # YOLO 검출 + 쿨다운 디바운싱 → pgm_output/ 저장
│   ├── run_demo.sh      # 자체 리허설용 (채점 shell과 동일 방식)
│   ├── data/            # 학습된 가중치 .bin 8개 + 기본 테스트 pgm
│   ├── pgm_output/      # PGM 저장 위치 (실행 시 생성, 비어 있음)
│   ├── FreeImage/
│   └── test.mp4         # 동작 확인용 (6,8,5)
├── yolo/
│   ├── best.pt          # 학습된 1클래스 digit 검출 모델
│   └── *.py             # YOLO 학습/라벨링 코드
└── mnist_train/
    ├── train_lenet.py       # LeNet 재학습 (.bin 생성)
    ├── convert_my_digits.py # 손글씨 사진 → pgm 변환
    └── my_digits_pgm/       # 변환된 6·8 학습용 pgm

(원본 MNIST 데이터셋, 원본 손글씨 사진, YOLO 학습 데이터/runs,
 기존 실험 pgm 등 시연에 불필요한 대용량 파일은 제출에서 제외)
```
