"""
Script to post-process NPZ files (raw model outputs) and evaluate them against GT
"""

import os
import numpy as np
import torch
import yaml
import cv2
from pathlib import Path
import argparse

from utils.util import post_process_yolo, non_max_suppression, compute_ap, wh2xy

# NPZ file output keys
OUTPUT_KEYS = [
    'tvmgen_default_ethos_n_main_62-1',  # P3 (80x80)
    'tvmgen_default_ethos_n_main_74-0',  # P4 (40x40)
    'tvmgen_default_ethos_n_main_96-0',  # P5 (20x20)
]

# Model output quantization parameters
OUTPUT_SCALES = [
    0.47186291217803955,  # P3
    0.46599841117858887,  # P4
    0.789106547832489     # P5
]

OUTPUT_ZERO_POINTS = [50, 59, 79]

# Model strides
MODEL_STRIDES = torch.tensor([8., 16., 32.])

# Class names
CLASS_NAMES = ['bicycle', 'bus', 'car', 'motorcycle', 'person', 'truck']
NUM_CLASSES = len(CLASS_NAMES)

# Post-processing thresholds
CONF_THRES = 0.001
IOU_THRES = 0.65

def load_and_process_npz(npz_path, num_classes=6):
    """
    Load NPZ file, post-process, and return final detection results
    """

    # Load NPZ file
    raw_outputs_np = np.load(npz_path)

    # Check and sort keys
    available_keys = list(raw_outputs_np.keys())

    # Use specified keys if present, otherwise sort by size
    if all(key in available_keys for key in OUTPUT_KEYS):
        output_keys = OUTPUT_KEYS
    else:
        output_keys = sorted(available_keys, key=lambda k: raw_outputs_np[k].size, reverse=True)
        print(f"Default output keys not found. Using keys: {output_keys[:3]}")

    num_channels = 16 + num_classes
    shapes = [
        (1, num_channels, 80, 80),
        (1, num_channels, 40, 40),
        (1, num_channels, 20, 20),
    ]

    # Dequantization and tensor conversion
    prediction_outputs = []
    for i, (key, shape) in enumerate(zip(output_keys[:3], shapes)):
        # Load INT8 numpy array
        int_tensor_np = raw_outputs_np[key]

        # Convert to Float32 and dequantize
        int_tensor = torch.from_numpy(int_tensor_np.astype(np.float32))
        scale = OUTPUT_SCALES[i]
        zero_point = OUTPUT_ZERO_POINTS[i]
        fp32_tensor = (int_tensor - zero_point) * scale

        # NHWC -> NCHW
        fp32_tensor = fp32_tensor.permute(0, 3, 1, 2).contiguous()
        prediction_outputs.append(fp32_tensor)

    # Post-processing - Bounding Box decoding
    decoded_outputs = post_process_yolo(prediction_outputs, MODEL_STRIDES)

    # NMS
    final_detections = non_max_suppression(
        decoded_outputs,
        confidence_threshold=CONF_THRES,
        iou_threshold=IOU_THRES
    )

    return final_detections[0]  # (N, 6) [x1, y1, x2, y2, conf, cls]


def load_yolo_labels(label_path, img_width, img_height):
    """
    Load YOLO format GT label file and convert to pixel coordinates
    """

    if not os.path.exists(label_path):
        return torch.empty(0).long(), torch.empty(0, 4)

    labels = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])

                # normalized -> pixel
                cx *= img_width
                cy *= img_height
                w *= img_width
                h *= img_height

                # xywh -> xyxy
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2

                labels.append([class_id, x1, y1, x2, y2])

    if len(labels) == 0:
        return torch.empty(0).long(), torch.empty(0, 4)

    labels_array = np.array(labels)
    classes = torch.from_numpy(labels_array[:, 0]).long()
    boxes = torch.from_numpy(labels_array[:, 1:5]).float()

    return classes, boxes


def box_iou(box1, box2):
    """Compute IoU"""
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou


def compute_metric(pred_boxes, pred_classes, gt_boxes, gt_classes, iou_v):
    """
    Compute metrics between predictions and GT
    """

    n_pred = len(pred_boxes)
    n_iou = len(iou_v)
    correct = torch.zeros(n_pred, n_iou, dtype=torch.bool)

    if len(gt_boxes) == 0 or n_pred == 0:
        return correct

    # Compute IoU
    iou = box_iou(gt_boxes, pred_boxes)  # (M, N)

    for i in range(n_iou):
        matches = []
        for gt_idx in range(len(gt_classes)):
            for pred_idx in range(n_pred):
                if iou[gt_idx, pred_idx] >= iou_v[i] and gt_classes[gt_idx] == pred_classes[pred_idx]:
                    matches.append([gt_idx, pred_idx, iou[gt_idx, pred_idx].item()])

        if len(matches) > 0:
            matches = np.array(matches)
            # Sort by IoU (descending)
            matches = matches[matches[:, 2].argsort()[::-1]]
            # Remove duplicate predictions
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # Remove duplicate ground truths
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

            correct[matches[:, 1].astype(int), i] = True

    return correct


def evaluate_npz_outputs(npz_dir, gt_labels_dir, gt_images_dir):
    """
    Post-process NPZ files and compare with GT to compute mAP
    """
    print(f"\n{'='*70}")
    print(f"NPZ Output Evaluation (Post-processing + mAP Computation)")
    print(f"{'='*70}")
    print(f"NPZ Directory: {npz_dir}")
    print(f"GT Labels: {gt_labels_dir}")
    print(f"GT Images: {gt_images_dir}")
    print(f"{'='*70}\n")

    # List of NPZ files
    npz_files = list(Path(npz_dir).glob('*.npz'))
    print(f"Found {len(npz_files)} NPZ files\n")

    if len(npz_files) == 0:
        print("No NPZ files found")
        return

    # Collect metrics
    metrics = []
    iou_v = torch.linspace(0.5, 0.95, 10)
    n_iou = iou_v.numel()

    print("Starting evaluation")

    processed = 0
    skipped = 0

    for npz_file in npz_files:
        filename = npz_file.stem.replace('_output', '')  # input1_output.npz -> input1

        try:
            # NPZ Post-processing
            detections = load_and_process_npz(str(npz_file), NUM_CLASSES)

            # Load GT
            # Find image file
            img_file = None
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
                candidate = Path(gt_images_dir) / f"{filename}{ext}"
                if candidate.exists():
                    img_file = candidate
                    break

            if not img_file:
                img_w, img_h = 640, 640
            else:
                img = cv2.imread(str(img_file))
                if img is None:
                    img_w, img_h = 640, 640
                else:
                    img_h, img_w = img.shape[:2]

            # Load GT labels
            gt_label_file = Path(gt_labels_dir) / f"{filename}.txt"
            gt_classes, gt_boxes = load_yolo_labels(str(gt_label_file), img_w, img_h)

            # Parse prediction data
            if len(detections) > 0:
                pred_boxes = detections[:, :4]
                pred_confs = detections[:, 4]
                pred_classes = detections[:, 5].long()
            else:
                pred_boxes = torch.empty(0, 4)
                pred_confs = torch.empty(0)
                pred_classes = torch.empty(0).long()

            # Compute metrics
            if len(pred_boxes) > 0:
                correct = compute_metric(pred_boxes, pred_classes, gt_boxes, gt_classes, iou_v)
            else:
                correct = torch.zeros(0, n_iou, dtype=torch.bool)

            # Save metrics
            metrics.append((
                correct,
                pred_confs if len(pred_confs) > 0 else torch.tensor([]),
                pred_classes.float() if len(pred_classes) > 0 else torch.tensor([]),
                gt_classes.float() if len(gt_classes) > 0 else torch.tensor([])
            ))

            processed += 1

        except Exception as e:
            print(f"Processing error: {filename}: {e}")
            skipped += 1
            continue

    print(f"Evaluation complete: {processed} processed, {skipped} skipped\n")

    if processed == 0:
        print("No files processed")
        return

    # Compute final mAP
    print(f"Computing metrics\n")

    # Aggregate data
    metrics_to_cat = list(zip(*metrics))
    tp_all = torch.cat([x for x in metrics_to_cat[0]], dim=0).cpu().numpy()
    conf_all = torch.cat([x for x in metrics_to_cat[1]], dim=0).cpu().numpy()
    pred_cls_all = torch.cat([x for x in metrics_to_cat[2]], dim=0).cpu().numpy()
    target_cls_all = torch.cat([x for x in metrics_to_cat[3]], dim=0).cpu().numpy()

    if tp_all.shape[0] == 0:
        print("No objects detected")
        return

    # Call compute_ap (plot=False)
    tp, fp, m_pre, m_rec, map50, mean_ap = compute_ap(
        tp_all, conf_all, pred_cls_all, target_cls_all,
        plot=False, names=CLASS_NAMES
    )

    # Compute per-class metrics
    eps = 1e-16

    # Sort by Confidence
    i = np.argsort(-conf_all)
    tp_sorted, conf_sorted, pred_cls_sorted = tp_all[i], conf_all[i], pred_cls_all[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls_all, return_counts=True)
    nc = unique_classes.shape[0]

    # Save per-class metrics
    class_metrics = {}
    p_per_class = np.zeros(nc)
    r_per_class = np.zeros(nc)
    ap_per_class = np.zeros((nc, tp_all.shape[1]))

    for ci, c in enumerate(unique_classes):
        i = pred_cls_sorted == c
        nl = nt[ci]  # GT count
        no = i.sum()  # Prediction count

        if no == 0 or nl == 0:
            continue

        # Accumulate TP/FP
        fpc = (1 - tp_sorted[i]).cumsum(0)
        tpc = tp_sorted[i].cumsum(0)

        # Recall
        recall = tpc / (nl + eps)

        # Precision
        precision = tpc / (tpc + fpc)

        # Compute AP (for all IoU thresholds)
        for j in range(tp_all.shape[1]):
            m_rec = np.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = np.concatenate(([1.0], precision[:, j], [0.0]))

            # Precision envelope
            m_pre = np.flip(np.maximum.accumulate(np.flip(m_pre)))

            # AP computation
            x = np.linspace(0, 1, 101)
            ap_per_class[ci, j] = np.trapz(np.interp(x, m_rec, m_pre), x)

        # Precision/recall at best F1
        f1 = 2 * precision * recall / (precision + recall + eps)
        i_f1 = f1.mean(1).argmax()
        p_per_class[ci] = precision[i_f1, 0]
        r_per_class[ci] = recall[i_f1, 0]

        class_metrics[int(c)] = {
            'precision': p_per_class[ci],
            'recall': r_per_class[ci],
            'ap50': ap_per_class[ci, 0],
            'ap': ap_per_class[ci].mean(),
            'n_gt': nl,
            'n_pred': no
        }
    
    mp = p_per_class.mean() if len(p_per_class) > 0 else 0.0
    mr = r_per_class.mean() if len(r_per_class) > 0 else 0.0
    map50_val = map50.mean()
    map_val = mean_ap.mean()

    # Print results
    print(f"{'='*70}")
    print(f"Evaluation Results")
    print(f"{'='*70}\n")

    print(f"Overall Performance:")
    print(f"  Precision    : {mp:.4f}") 
    print(f"  Recall       : {mr:.4f}") 
    print(f"  mAP@0.5      : {map50_val:.4f}")
    print(f"  mAP@0.5:0.95 : {map_val:.4f}")

    print(f"\n{'='*70}")
    print(f"Per-class Performance:")
    print(f"{'='*70}")

    # Per-class GT counts
    unique_gt, counts_gt = np.unique(target_cls_all, return_counts=True)
    gt_counts_dict = dict(zip(unique_gt.astype(int), counts_gt))

    # Per-class prediction counts
    unique_pred, counts_pred = np.unique(pred_cls_all, return_counts=True)
    pred_counts_dict = dict(zip(unique_pred.astype(int), counts_pred))

    print(f"{'Class':<12} {'Instances':>10} {'P':>8} {'R':>8} {'mAP@.5':>10} {'mAP@.5:.95':>12}")
    print(f"{'-'*70}")

    for class_id in range(NUM_CLASSES):
        class_name = CLASS_NAMES[class_id]
        gt_count = gt_counts_dict.get(class_id, 0)

        if class_id in class_metrics:
            metrics = class_metrics[class_id]
            precision = metrics['precision']
            recall = metrics['recall']
            ap50 = metrics['ap50']
            ap = metrics['ap']
            print(f"{class_name:<12} {gt_count:>10} {precision:>8.3f} {recall:>8.3f} {ap50:>10.4f} {ap:>12.4f}")
        else:
            print(f"{class_name:<12} {gt_count:>10} {'N/A':>8} {'N/A':>8} {'N/A':>10} {'N/A':>12}")

    print(f"{'-'*70}")
    print(f"{'All':<12} {len(target_cls_all):>10} {mp:>8.3f} {mr:>8.3f} {map50_val:>10.4f} {map_val:>12.4f}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NPZ output evaluation script")
    parser.add_argument('--npz-dir', type=str, default='outputs',
                       help='NPZ file directory')
    parser.add_argument('--gt-labels', type=str, default='input_label',
                       help='Ground Truth label directory')
    parser.add_argument('--gt-images', type=str, default='input_jpg',
                       help='Ground Truth image directory')

    args = parser.parse_args()

    evaluate_npz_outputs(args.npz_dir, args.gt_labels, args.gt_images)