# NPZ Output Evaluation Guide

YOLOv11 모델의 NPZ 출력 파일을 후처리하고 Ground Truth와 비교하여 mAP를 계산하는 평가 스크립트입니다.

## 개요

이 스크립트는 다음 작업을 수행합니다:
1. NPZ 파일 로드 (INT8 양자화된 모델 출력)
2. 역양자화 (INT8 → FP32)
3. YOLO 후처리 (Bounding Box 디코딩)
4. NMS (Non-Maximum Suppression)
5. Ground Truth와 비교
6. mAP 계산 및 출력

## 디렉토리 구조

```
yolo11-optimization/
├── eval/
│   ├── eval_output_npz.py    # 평가 스크립트
│   └── README.md             # 이 가이드
├── utils/
│   ├── __init__.py
│   └── util.py               # 후처리 함수들
└── output/
    ├── npz/                  # 모델 출력 NPZ 파일
    │   ├── input1_output.npz
    │   ├── input2_output.npz
    │   └── ...
    ├── labels/               # Ground Truth 레이블
    │   ├── input1.txt
    │   ├── input2.txt
    │   └── ...
    └── images/               # 원본 이미지
        ├── input1.jpg
        ├── input2.jpg
        └── ...
```

## 사용법

### 기본 실행

`yolo11-optimization` 디렉토리에서 실행:

```bash
python eval/eval_output_npz.py \
    --npz-dir output/npz \
    --gt-labels output/labels \
    --gt-images output/images
```

### 인자 설명

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--npz-dir` | `outputs` | NPZ 파일이 있는 디렉토리 |
| `--gt-labels` | `input_label` | YOLO 형식의 GT 레이블 디렉토리 |
| `--gt-images` | `input_jpg` | 원본 이미지 디렉토리 (크기 확인용) |

## 파일 형식

### NPZ 파일

모델 출력을 저장한 NumPy 압축 파일입니다.

- 파일명: `{name}_output.npz`
- 3개의 feature map 포함:
  - P3: 80x80 (stride 8)
  - P4: 40x40 (stride 16)
  - P5: 20x20 (stride 32)

### GT 레이블 (.txt)

YOLO 형식의 레이블 파일입니다.

```
<class_id> <cx> <cy> <w> <h>
```

- 좌표는 0~1 사이의 정규화된 값
- 예: `2 0.5 0.5 0.3 0.4` (class 2, 중심 50%, 너비 30%, 높이 40%)

## 설정 값

스크립트 내에서 다음 값들을 수정할 수 있습니다:

### 클래스 정보

```python
CLASS_NAMES = ['bicycle', 'bus', 'car', 'motorcycle', 'person', 'truck']
NUM_CLASSES = 6
```

### 양자화 파라미터

```python
OUTPUT_SCALES = [0.47186291217803955, 0.46599841117858887, 0.789106547832489]
OUTPUT_ZERO_POINTS = [50, 59, 79]
```

### 후처리 임계값

```python
CONF_THRES = 0.001   # Confidence threshold
IOU_THRES = 0.65     # NMS IoU threshold
```

## 출력 결과

### 전체 성능

```
Overall Performance:
  Precision    : 0.9423
  Recall       : 0.8813
  mAP@0.5      : 0.9266
  mAP@0.5:0.95 : 0.6509
```

### 클래스별 성능

```
======================================================================
Per-class Performance:
======================================================================
Class         Instances        P        R     mAP@.5   mAP@.5:.95
----------------------------------------------------------------------
bicycle             223    0.971    0.906     0.9706       0.6826
bus                 245    0.975    0.955     0.9832       0.6816
car                 246    0.970    0.931     0.9703       0.6993
motorcycle          200    0.970    0.975     0.9897       0.7832
person             1304    0.782    0.550     0.6596       0.3286
truck               206    0.985    0.971     0.9864       0.7298
----------------------------------------------------------------------
All                2424    0.942    0.881     0.9266       0.6509

======================================================================
```

## 동작 흐름

```
1. NPZ 로드
   └─ np.load(npz_path)

2. 역양자화
   └─ fp32 = (int8 - zero_point) * scale

3. 형식 변환
   └─ NHWC → NCHW

4. Box 디코딩
   └─ post_process_yolo()
   └─ anchor 생성 → offset 적용 → stride 곱셈

5. NMS
   └─ non_max_suppression()
   └─ confidence 필터링 → torchvision.ops.nms()

6. GT 로드
   └─ load_yolo_labels()
   └─ normalized xywh → pixel xyxy

7. 메트릭 계산
   └─ compute_metric() → IoU 0.5~0.95 (10단계)

8. mAP 계산
   └─ compute_ap()
   └─ Precision-Recall 커브 → 101-point 보간
```

### 이미지를 찾을 수 없음

이미지 파일이 없어도 평가는 진행됩니다. 기본 크기 640x640을 사용합니다.
정확한 평가를 위해서는 원본 이미지 크기가 필요합니다.

### NPZ 키를 찾을 수 없음

NPZ 파일의 출력 키가 기본값과 다른 경우, 자동으로 크기 순으로 정렬하여 사용합니다.
경고 메시지가 출력됩니다.
