import torch
import numpy as np
import cv2
import argparse
import os
from pathlib import Path
import sys

# Import post-processing and NMS functions
from utils.util import post_process_yolo, non_max_suppression

# Define class names (needs modification to match actual dataset)
CLASS_NAMES = [
    'bicycle', 'bus', 'car', 'motorcycle', 'person', 'truck', 'dummy', 'dummy',
    'dummy', 'dummy', 'dummy', 'dummy', 'dummy', 'dummy', 'dummy', 'dummy'
    ]

# Model stride values
STRIDES = torch.tensor([8., 16., 32.])

# IMPORTANT: Must be replaced with actual Scale and Zero-Point values of the model
# Output order is based on P3(80x80), P4(40x40), P5(20x20)
OUTPUT_SCALES = [
    0.47186291217803955,  # output_0 (P3)
    0.46599841117858887,  # output_1 (P4)
    0.789106547832489     # output_2 (P5)
] 
OUTPUT_ZERO_POINTS = [
    50,
    59,
    79
] 
OUTPUT_DTYPE = np.int8

def run_postprocessing(npz_path, image_path, output_path, conf_thres, iou_thres, nc):
    """
    Process a single image and its corresponding NPZ file.
    """

    # Load NPZ file
    try:
        raw_outputs_np = np.load(npz_path)
    except Exception as e:
        print(f"[Error] Failed to load {npz_path}: {e}")
        return

    output_keys = [
        'tvmgen_default_ethos_n_main_62-1',
        'tvmgen_default_ethos_n_main_74-0',
        'tvmgen_default_ethos_n_main_96-0',
    ]

    num_channels = 16 + nc 

    shapes = [
        (1, num_channels, 80, 80),
        (1, num_channels, 40, 40),
        (1, num_channels, 20, 20), 
    ]

    # Check NPZ file keys
    if not all(key in raw_outputs_np for key in output_keys):
        output_keys = list(raw_outputs_np.keys())
        output_keys.sort(key=lambda k: raw_outputs_np[k].size, reverse=True)

    prediction_outputs = []
    for i, (key, shape) in enumerate(zip(output_keys, shapes)):
        int_tensor_np = raw_outputs_np[key]
        int_tensor = torch.from_numpy(int_tensor_np.astype(np.float32))

        scale = OUTPUT_SCALES[i]
        zero_point = OUTPUT_ZERO_POINTS[i]
        fp32_tensor = (int_tensor - zero_point) * scale

        fp32_tensor = fp32_tensor.permute(0, 3, 1, 2)
        prediction_outputs.append(fp32_tensor.contiguous())

    # Post-process
    decoded_outputs = post_process_yolo(prediction_outputs, STRIDES)
    final_detections = non_max_suppression(decoded_outputs, confidence_threshold=conf_thres, iou_threshold=iou_thres)

    # Visualization
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[Error] Could not read image: {image_path}")
        return

    img_h, img_w = image.shape[:2]
    detections = final_detections[0]

    object_count = 0
    if detections is not None and len(detections) > 0:
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            if 0 <= int(cls_id) < len(CLASS_NAMES) -1:
                object_count += 1
                # Draw bounding box
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                label = f"{CLASS_NAMES[int(cls_id)]} {conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 2
                
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
                
                text_x = int(x1)
                text_y = int(y1) - 10
                if text_y < text_h + 5:
                    text_y = int(y1) + text_h + 5
                
                cv2.rectangle(image, (text_x, text_y - text_h - baseline), (text_x + text_w, text_y + baseline), (0, 255, 0), -1)
                cv2.putText(image, label, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    # Save result
    cv2.imwrite(str(output_path), image)
    print(f"Saved: {output_path.name} (Detected: {object_count})")


def main():
    parser = argparse.ArgumentParser(description="Batch YOLOv11 TVM output post-processing script")
    parser.add_argument('--npz-dir', type=str, default='outputs', help="Directory containing .npz files")
    parser.add_argument('--input-dir', type=str, default='input_jpg', help="Directory containing input images")
    parser.add_argument('--output-dir', type=str, default='detection_result', help="Directory to save result images")
    parser.add_argument('--conf-thres', type=float, default=0.01, help="Confidence threshold")
    parser.add_argument('--iou-thres', type=float, default=0.65, help="IoU threshold")
    parser.add_argument('--num-classes', type=int, default=16, help="Number of classes including padding")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    npz_dir = Path(args.npz_dir)
    output_dir = Path(args.output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_dir.glob(ext)))
    
    # Sort for consistent processing order
    image_files.sort()

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(image_files)} images. Starting batch processing...")
    print("-" * 50)

    success_count = 0
    
    for img_path in image_files:
        # Construct corresponding NPZ filename
        # Assumption: input1.jpg -> input1_output.npz
        stem = img_path.stem  # 'input1'
        npz_filename = f"{stem}_output.npz" # 'input1_output.npz'
        npz_path = npz_dir / npz_filename
        
        # Construct output filename
        output_filename = f"result_{img_path.name}"
        output_path = output_dir / output_filename

        if not npz_path.exists():
            print(f"[Skip] NPZ file not found for {img_path.name} (Expected: {npz_filename})")
            continue

        # Run processing
        try:
            run_postprocessing(
                npz_path, 
                img_path, 
                output_path, 
                args.conf_thres, 
                args.iou_thres, 
                args.num_classes
            )
            success_count += 1
        except Exception as e:
            print(f"[Error] Failed processing {img_path.name}: {e}")

    print("-" * 50)
    print(f"Batch processing complete. {success_count}/{len(image_files)} images processed.")
    print(f"Results saved in '{output_dir}'")

if __name__ == "__main__":
    main()