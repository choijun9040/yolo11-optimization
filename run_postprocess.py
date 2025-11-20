import torch
import numpy as np
import cv2
import argparse

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
    0.789106547832489   # output_2 (P5)
] # Replace with actual Scale values
OUTPUT_ZERO_POINTS = [
    50,
    59,
    79
] # Replace with actual Zero-Point values
OUTPUT_DTYPE = np.int8 # Replace with actual output type (int8 or uint8)

def run_postprocessing(npz_path, image_path, output_path, conf_thres, iou_thres, nc):
    """
    Main function to load TVM output, post-process it, and visualize the results.
    nc: Number of classes including padding (e.g., 16)
    """
    print(f"Loading raw output data from '{npz_path}'")

    # Load NPZ file
    raw_outputs_np = np.load(npz_path)

    output_keys = [
        'tvmgen_default_ethos_n_main_62-1',
        'tvmgen_default_ethos_n_main_74-0',
        'tvmgen_default_ethos_n_main_96-0',
    ]

    print("Converting Numpy arrays to PyTorch tensors, dequantizing, and reshaping")

    num_channels = 16 + nc 

    shapes = [
        (1, num_channels, 80, 80),
        (1, num_channels, 40, 40),
        (1, num_channels, 20, 20), 
    ]

    # Check NPZ file keys
    if not all(key in raw_outputs_np for key in output_keys):
        print("Warning: Specified output keys not found in NPZ file. Using keys within the file")
        output_keys = list(raw_outputs_np.keys())
        output_keys.sort(key=lambda k: raw_outputs_np[k].size, reverse=True)
        print(f"Keys to be used (sorted by size): {output_keys}")


    prediction_outputs = []
    for i, (key, shape) in enumerate(zip(output_keys, shapes)):
        # Load INT8/UINT8 Numpy array from NPZ
        int_tensor_np = raw_outputs_np[key]

        # Convert to PyTorch tensor (Change type to float32 for calculation)
        int_tensor = torch.from_numpy(int_tensor_np.astype(np.float32))

        # Perform Dequantization
        scale = OUTPUT_SCALES[i]
        zero_point = OUTPUT_ZERO_POINTS[i]
        fp32_tensor = (int_tensor - zero_point) * scale
        print(f" - '{key}': Dequantization complete (scale={scale}, zero_point={zero_point})")

        # Final Reshape and add to list
        fp32_tensor = fp32_tensor.permute(0, 3, 1, 2)
        prediction_outputs.append(fp32_tensor.contiguous())

    print("Running post_process_yolo for Bounding Box decoding")
    decoded_outputs = post_process_yolo(prediction_outputs, STRIDES)


    print("Running NMS to remove overlapping Bounding Boxes")
    final_detections = non_max_suppression(decoded_outputs, confidence_threshold=conf_thres, iou_threshold=iou_thres)

    print(f"Drawing detection results on '{image_path}'")
    image = cv2.imread(image_path)
    # Get image height and width
    img_h, img_w = image.shape[:2]

    detections = final_detections[0] # Use the first result in the batch

    if detections is not None and len(detections) > 0:
        print(f"{len(detections)} objects detected")
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            # Check if Class ID is valid (excluding padding classes)
            if 0 <= int(cls_id) < len(CLASS_NAMES) -1: # Do not draw the last dummy class
                # Draw bounding box
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                label = f"{CLASS_NAMES[int(cls_id)]} {conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 2
                
                # Calculate text size
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
                
                # Calculate starting y-coordinate for text drawing
                # Draws above the bounding box by default, but draws inside if it exceeds the top of the image
                text_x = int(x1)
                text_y = int(y1) - 10 # Default: 10 pixels above the bounding box
                
                # If text exceeds the image top
                if text_y < text_h + 5: # If smaller than text height + slight margin
                    text_y = int(y1) + text_h + 5 # Adjust to inside bounding box (top + text height + margin)
                
                # Draw text background box
                # Calculate text background box coordinates
                bg_x1 = text_x
                bg_y1 = text_y - text_h - baseline
                bg_x2 = text_x + text_w
                bg_y2 = text_y + baseline
                
                # Ensure background box does not exceed image boundaries (optional but for safety)
                bg_x1 = max(0, bg_x1)
                bg_y1 = max(0, bg_y1)
                bg_x2 = min(img_w, bg_x2)
                bg_y2 = min(img_h, bg_y2)

                # Draw background box
                cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 255, 0), -1) # -1 fills the shape
                
                # Draw text
                cv2.putText(image, label, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)


    else:
        print("No objects detected")

    # Save result image
    cv2.imwrite(output_path, image)
    print(f"Final result saved to '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv11 TVM output post-processing script")
    parser.add_argument('--npz', type=str, default='outputs/input10_output.npz', help="Path to the rtvm output .npz file")
    parser.add_argument('--image', type=str, default='input_jpg/input10.jpg', help="Path to the original input image for visualization")
    parser.add_argument('--output-image', type=str, default='detection_result/detection_result_10.jpg', help="Path to save the output image with detections")
    parser.add_argument('--conf-thres', type=float, default=0.01, help="Confidence threshold for filtering detections")
    parser.add_argument('--iou-thres', type=float, default=0.65, help="IoU threshold for Non-Maximum Suppression")
    parser.add_argument('--num-classes', type=int, default=16, help="Number of classes in the model (including padding)")


    args = parser.parse_args()

    # Pass the number of classes including padding to the run_postprocessing function
    run_postprocessing(args.npz, args.image, args.output_image, args.conf_thres, args.iou_thres, args.num_classes)