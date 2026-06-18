#!/bin/bash
# =====================================================================
#  run_demo.sh  -  영상 한 개를 넣으면 전 과정을 한 번에 실행
#  실행: mnistCUDNN 폴더 안에서
#     cd ~/Desktop/mnist_ocr_project/mnistCUDNN
#     ./run_demo.sh <영상파일>
#
#  과정(한 번에):
#    0) (옵션) 보드 성능 최대화
#    1) 영상 -> YOLO 검출 + 디바운싱 -> PGM 저장 (pgm_output/)
#    2) pgm_output/ 의 앞 N개를 mnistCUDNN image= 모드로 분류
#    3) answers 배열(정답)과 비교 -> 정답/추론시간/정확도 출력
# =====================================================================

PGM_DIR="pgm_output"
EXE="./mnistCUDNN"

# ---------------------------------------------------------------------
# [정답 배열] 영상의 손글씨 숫자를 등장 순서대로 입력 (채점자가 수정)
#   예) 영상이 6,6,6,6,1,5,6,8,6,3,8,6 이면:  answers=(6 6 6 6 1 5 6 8 6 3 8 6)
# ---------------------------------------------------------------------
answers=(6 8 5)

VIDEO=${1:-test.mp4}

# CUDA 런타임 경로
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "==================================================="
echo " 입력 영상 : $VIDEO"
echo " 정답      : ${answers[*]}  (${#answers[@]}개)"
echo "==================================================="

# ---------------------------------------------------------------------
# 0) 보드 성능 최대화 (sudo 권한 있을 때만, 첫 이미지 추론 8배 가속)
# ---------------------------------------------------------------------
if command -v jetson_clocks >/dev/null 2>&1; then
    sudo -n nvpmodel -m 0 >/dev/null 2>&1 && sudo -n jetson_clocks >/dev/null 2>&1 \
        && echo "[성능 모드 고정 완료]" \
        || echo "[성능 모드 설정 건너뜀 - 필요시 수동: sudo nvpmodel -m 0 && sudo jetson_clocks]"
fi

# ---------------------------------------------------------------------
# 1) YOLO 검출 + PGM 생성 (pgm_output/)
# ---------------------------------------------------------------------
echo ""
echo "----- 1) YOLO 검출 + PGM 생성 -----"
python3 detect_yolo.py "$VIDEO"

NUM_PGM=$(ls ${PGM_DIR}/*.pgm 2>/dev/null | wc -l)
echo "저장된 PGM 개수 : $NUM_PGM   (정답 개수 : ${#answers[@]})"
if [ "$NUM_PGM" -ne "${#answers[@]}" ]; then
    echo "  ※ 경고: PGM 개수와 정답 개수가 다릅니다!"
    echo "    -> detect_yolo.py 의 COOLDOWN_SEC 조정 후 다시 실행하세요."
    echo "       (많이 저장=쪼개짐: 값 늘리기 / 적게 저장=합쳐짐: 값 줄이기)"
fi

# ---------------------------------------------------------------------
# 2) MNIST 분류 (image= 단일 모드) + 정답 비교
# ---------------------------------------------------------------------
echo ""
echo "----- 2) MNIST 분류 + 채점 -----"
total=0
correct=0
total_infer_ms=0

for img in $(find "$PGM_DIR" -name "*.pgm" | sort -V | head -n ${#answers[@]})
do
    output=$(${EXE} image="$img")
    digit=$(echo "$output" | grep "Result of classification" | awk '{print $4}')
    infer_ms=$(echo "$output" | grep "Inference time:" | awk '{print $3}')
    gt=${answers[$total]}

    if [[ "$digit" == "$gt" ]]; then
        mark="O"; correct=$((correct+1))
    else
        mark="X"
    fi
    printf "  [%2d] 정답=%s  추론=%s  (%s ms)  ->  %s\n" "$((total+1))" "$gt" "$digit" "$infer_ms" "$mark"

    if [[ "$infer_ms" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        total_infer_ms=$(echo "$total_infer_ms + $infer_ms" | bc -l)
    fi
    total=$((total+1))
done

echo ""
echo "============= SUMMARY ============="
echo " Total Images          : $total"
echo " Correct Predictions   : $correct / $total"
if [ "$total" -gt 0 ]; then
    printf " Total Inference Time  : %.3f ms\n" "$total_infer_ms"
    printf " Average / Image       : %.3f ms\n" "$(echo "$total_infer_ms / $total" | bc -l)"
fi
echo "==================================="
