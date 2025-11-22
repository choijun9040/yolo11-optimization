# NPZ Output Evaluation Guide

This evaluation script post-processes NPZ output files from YOLOv11 model and computes mAP by comparing with Ground Truth.

## Overview

This script performs the following tasks:
1. Load NPZ files (INT8 quantized model output)
2. Dequantization (INT8 → FP32)
3. YOLO post-processing (Bounding Box decoding)
4. NMS (Non-Maximum Suppression)
5. Comparison with Ground Truth
6. mAP computation and output

## Directory Structure

```
yolo11-optimization/
├── eval/
│   ├── eval_output_npz.py    # Evaluation script
│   └── README.md             # This guide
├── utils/
│   ├── __init__.py
│   └── util.py               # Post-processing functions
└── output/
    ├── npz/                  # Model output NPZ files
    │   ├── input1_output.npz
    │   ├── input2_output.npz
    │   └── ...
    ├── labels/               # Ground Truth labels
    │   ├── input1.txt
    │   ├── input2.txt
    │   └── ...
    └── images/               # Original images
        ├── input1.jpg
        ├── input2.jpg
        └── ...
```

## Usage

### Basic Execution

Run from `yolo11-optimization` directory:

```bash
python eval/eval_output_npz.py \
    --npz-dir output/npz \
    --gt-labels output/labels \
    --gt-images output/images
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--npz-dir` | `outputs` | Directory containing NPZ files |
| `--gt-labels` | `input_label` | Directory with YOLO format GT labels |
| `--gt-images` | `input_jpg` | Directory with original images (for size check) |

## File Formats

### NPZ Files

NumPy compressed files storing model outputs.

- Filename: `{name}_output.npz`
- Contains 3 feature maps:
  - P3: 80x80 (stride 8)
  - P4: 40x40 (stride 16)
  - P5: 20x20 (stride 32)

### GT Labels (.txt)

YOLO format label files.

```
<class_id> <cx> <cy> <w> <h>
```

- Coordinates are normalized values between 0~1
- Example: `2 0.5 0.5 0.3 0.4` (class 2, center 50%, width 30%, height 40%)

## Configuration

The following values can be modified in the script:

### Class Information

```python
CLASS_NAMES = ['bicycle', 'bus', 'car', 'motorcycle', 'person', 'truck']
NUM_CLASSES = 6
```

### Quantization Parameters

```python
OUTPUT_SCALES = [0.47186291217803955, 0.46599841117858887, 0.789106547832489]
OUTPUT_ZERO_POINTS = [50, 59, 79]
```

### Post-processing Thresholds

```python
CONF_THRES = 0.001   # Confidence threshold
IOU_THRES = 0.65     # NMS IoU threshold
```

## Output Results

### Overall Performance

```
Overall Performance:
  Precision    : 0.7234
  Recall       : 0.6891
  mAP@0.5      : 0.7012
  mAP@0.5:0.95 : 0.4523
```

### Per-class Performance

```
Class        Instances        P        R     mAP@.5   mAP@.5:.95
----------------------------------------------------------------------
bicycle            120    0.723    0.689     0.7012       0.4523
bus                 85    0.812    0.745     0.7891       0.5234
car                450    0.756    0.712     0.7345       0.4812
...
----------------------------------------------------------------------
All               1000    0.723    0.689     0.7012       0.4523
```

## Processing Flow

```
1. Load NPZ
   └─ np.load(npz_path)

2. Dequantization
   └─ fp32 = (int8 - zero_point) * scale

3. Format Conversion
   └─ NHWC → NCHW

4. Box Decoding
   └─ post_process_yolo()
   └─ anchor generation → offset application → stride multiplication

5. NMS
   └─ non_max_suppression()
   └─ confidence filtering → torchvision.ops.nms()

6. Load GT
   └─ load_yolo_labels()
   └─ normalized xywh → pixel xyxy

7. Metric Computation
   └─ compute_metric() → IoU 0.5~0.95 (10 steps)

8. mAP Computation
   └─ compute_ap()
   └─ Precision-Recall curve → 101-point interpolation
```

## Dependencies

- Python 3.x
- numpy
- torch
- torchvision
- opencv-python (cv2)
- PyYAML

Installation:
```bash
pip install numpy torch torchvision opencv-python pyyaml
```

## Troubleshooting

### ModuleNotFoundError: No module named 'utils'

Run from `yolo11-optimization` directory:

```bash
cd /path/to/yolo11-optimization
python eval/eval_output_npz.py ...
```

### Image not found

Evaluation proceeds even without image files. Default size 640x640 is used.
For accurate evaluation, original image size is required.

### NPZ keys not found

If NPZ file output keys differ from defaults, they are automatically sorted by size and used.
A warning message will be displayed.
