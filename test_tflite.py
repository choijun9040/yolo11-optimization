import numpy as np
import torch
import tqdm
import yaml
import argparse
import os
from PIL import Image

# Import TFLite Interpreter (Select one)
# import tensorflow as tf # If full TensorFlow is installed
import tflite_runtime.interpreter as tflite # If only TFLite runtime is installed

# Reuse existing utility functions
from utils.dataset import Dataset # Can reuse some image preprocessing parts
from utils.util import post_process_yolo, non_max_suppression, compute_ap, wh2xy, compute_metric # compute_ap etc. might use wh2xy
from nets import nn # Model structure definition is not needed, but stride values can be retrieved

# Settings
TFLITE_MODEL_PATH = 'best/best_full_integer_quant.tflite' # Path to the converted TFLite model
DATA_DIR = '../Dataset/MyFirstProject' # Path to the dataset to evaluate
VAL_FILE = 'test.txt' # List of evaluation data
INPUT_SIZE = 640 # Model input size (Check value used during training/TFLite conversion)
BATCH_SIZE = 1 # TFLite usually infers with batch size 1
CONF_THRES = 0.001 # Initial threshold to match compute_ap (Minimize filtering before NMS)
IOU_THRES = 0.65  # NMS IoU threshold
NUM_CLASSES = 16  # Number of classes including padding (Verify same as args.yaml)

# Load class names (for compute_ap)
try:
    with open('utils/args.yaml', errors='ignore') as f:
        PARAMS = yaml.safe_load(f)
        CLASS_NAMES = PARAMS['names']
        # Check if configured NUM_CLASSES matches the number of classes in args.yaml
        if len(CLASS_NAMES) != NUM_CLASSES:
             print(f"Error: NUM_CLASSES in script ({NUM_CLASSES}) does not match class count in args.yaml ({len(CLASS_NAMES)})")
             exit()
except FileNotFoundError:
    print("Error: 'utils/args.yaml' file not found")
    exit()
except Exception as e:
    print(f"Error: An error occurred while loading 'utils/args.yaml': {e}")
    exit()


# Model Stride values
MODEL_STRIDES = torch.tensor([8., 16., 32.])


def evaluate_tflite():
    """Main function to evaluate mAP of the TFLite model"""

    # Load TFLite Interpreter
    print(f"Loading TFLite model: {TFLITE_MODEL_PATH}")
    try:
        interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Error: Failed to load TFLite model ({TFLITE_MODEL_PATH}). Check file path and the file itself: {e}")
        return

    # Get input/output details
    try:
        input_details = interpreter.get_input_details()[0] # Assuming 1 input
        output_details = interpreter.get_output_details() # 3 outputs
    except Exception as e:
        print(f"Error: Failed to get TFLite model I/O details. Model file might be corrupted or invalid: {e}")
        return

    # Check input type
    input_dtype = input_details['dtype']
    print(f" - Input Type: {input_dtype}")
    # Check input quantization parameters (Needed for INT8 conversion)
    input_quant_params = input_details['quantization_parameters']
    input_scale = input_quant_params['scales'][0] if len(input_quant_params['scales']) > 0 else 1.0
    input_zero_point = input_quant_params['zero_points'][0] if len(input_quant_params['zero_points']) > 0 else 0
    print(f" - Input Scale: {input_scale}")
    print(f" - Input Zero Point: {input_zero_point}")

    # Extract output info (Scale, Zero-Point) and sort order (P3, P4, P5)
    try:
        # Reconfirm logic to sort in P3(80) > P4(40) > P5(20) order
        output_details_sorted = sorted(output_details,
                                       key=lambda d: d['shape'][1], # Sort by H (index 1) of NHWC
                                       reverse=True) # Descending order of H (P3, P4, P5)
        
        # Extract indices and parameters in sorted order
        output_indices_sorted = [d['index'] for d in output_details_sorted]
        output_scales = [d['quantization_parameters']['scales'][0] for d in output_details_sorted]
        output_zero_points = [d['quantization_parameters']['zero_points'][0] for d in output_details_sorted]
        output_shapes = [d['shape'] for d in output_details_sorted]
        output_dtypes = [d['dtype'] for d in output_details_sorted]
        
    except Exception as e:
        print(f"Error: Failed to extract TFLite model output quantization info: {e}")
        return

    print(" - Output Scale:", output_scales)
    print(" - Output Zero Points:", output_zero_points)
    print(" - Output Shapes (NHWC):", output_shapes)
    print(" - Output Dtypes:", output_dtypes)

    # Prepare Dataset
    filenames = []
    print(f"Loading image list from '{DATA_DIR}/{VAL_FILE}'")
    image_folder = os.path.join(DATA_DIR, 'images', 'test') # Assuming 'test' folder usage

    try:
        with open(f'{DATA_DIR}/{VAL_FILE}') as f:
            for line in f.readlines():
                # Assuming line.strip() contains only image filename (e.g., '0001.jpg')
                # Or extract filename matching test.txt content format
                filename = os.path.basename(line.strip()) # Extract filename only if path is included
                if not filename: continue

                # Generate full path
                full_img_path = os.path.join(image_folder, filename)

                if os.path.exists(full_img_path):
                     filenames.append(full_img_path)
                else:
                     # Attempt to find image based on label file path (Refer to Dataset class logic)
                     label_file_base = os.path.splitext(filename)[0]
                     potential_img_path_from_label = os.path.join(DATA_DIR, 'images', 'test', label_file_base + '.jpg') # Assuming .jpg
                     if os.path.exists(potential_img_path_from_label):
                         filenames.append(potential_img_path_from_label)
                     else:
                         print(f"Warning: File not found '{full_img_path}' or '{potential_img_path_from_label}'. Check path logic and content of '{VAL_FILE}'")
    except FileNotFoundError:
        print(f"Error: File '{DATA_DIR}/{VAL_FILE}' not found")
        return
    except Exception as e:
        print(f"Error: Error processing file '{DATA_DIR}/{VAL_FILE}': {e}")
        return


    if not filenames:
        print(f"Error: No valid image files found in '{DATA_DIR}/{VAL_FILE}'. Check path and file content")
        return

    print(f"Loaded {len(filenames)} evaluation image file paths")

    # Load target (ground truth) using Dataset class
    try:
        eval_dataset = Dataset(filenames, INPUT_SIZE, PARAMS, augment=False)
    except Exception as e:
        print(f"Error: Failed to create Dataset object: {e}")
        return

    # Evaluation Loop
    metrics = [] # List to store results for final evaluation
    p_bar = tqdm.tqdm(range(len(eval_dataset)), desc='Evaluating TFLite Model')

    for img_idx in p_bar:
        try:
            # Get sample and target from dataset
            sample_tensor, target_cls, target_box, _ = eval_dataset[img_idx]
            cls = target_cls
            box = target_box

            # Dataset output (NCHW, Float, 0-255) -> NHWC, Float, 0-255
            input_data_np_f32_255range = sample_tensor.squeeze(0).permute(1, 2, 0).numpy()

            # Normalize to 0.0 ~ 1.0 range
            input_data_np_f32_1range = input_data_np_f32_255range.astype(np.float32) / 255.0

            # Get scale, zero_point, dtype of TFLite model input
            input_scale = input_details['quantization_parameters']['scales'][0]
            input_zero_point = input_details['quantization_parameters']['zero_points'][0]

            # Apply quantization formula (FP32 0.0-1.0 range -> INT8)
            input_data_quantized = (input_data_np_f32_1range / input_scale) + input_zero_point
            
            # Round and Type Casting
            input_data_quantized = np.round(input_data_quantized)
            if input_dtype == np.int8:
                input_data_quantized = np.clip(input_data_quantized, -128, 127)
            elif input_dtype == np.uint8:
                input_data_quantized = np.clip(input_data_quantized, 0, 255)
            input_data = input_data_quantized.astype(input_dtype)

            # Add batch dimension
            input_data = np.expand_dims(input_data, axis=0)

            # Run TFLite Inference
            interpreter.set_tensor(input_details['index'], input_data)
            interpreter.invoke()

            # Get output tensors (In sorted order)
            raw_outputs_int = [interpreter.get_tensor(i) for i in output_indices_sorted]

            # Dequantization and PyTorch Tensor Conversion
            prediction_outputs_torch = []
            for i, int_tensor_np in enumerate(raw_outputs_int):
                # INT8/UINT8 -> Float32 Conversion
                dequantized_tensor = (int_tensor_np.astype(np.float32) - output_zero_points[i]) * output_scales[i]
                fp32_tensor = torch.from_numpy(dequantized_tensor)

                # NHWC -> NCHW Conversion
                n, h, w, c = output_shapes[i]
                fp32_tensor = fp32_tensor.reshape(n, h, w, c).permute(0, 3, 1, 2)
                prediction_outputs_torch.append(fp32_tensor)

            # Run Post-processing
            decoded_outputs = post_process_yolo(prediction_outputs_torch, MODEL_STRIDES)
            final_detections = non_max_suppression(decoded_outputs, confidence_threshold=CONF_THRES, iou_threshold=IOU_THRES)
            output = final_detections[0] # Result of batch 0

            # Prepare Metric Calculation
            scale = torch.tensor((INPUT_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_SIZE)) # Assume xyxy order
            iou_v = torch.linspace(start=0.5, end=0.95, steps=10)
            n_iou = iou_v.numel()
            metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool)

            conf_tensor_to_add = torch.tensor([]) # Default empty tensor (1D)
            pred_cls_tensor_to_add = torch.tensor([]) # Default empty tensor (1D)

            if output.shape[0] > 0: # If objects detected
                conf_tensor_to_add = output[:, 4].flatten() # Explicit 1D conversion with flatten()
                pred_cls_tensor_to_add = output[:, 5] # Class is already 1D or flatten unnecessary

                if cls.shape[0]: # If ground truth exists, compute metric
                    target_box_xyxy = wh2xy(box) * scale
                    target = torch.cat(tensors=(cls, target_box_xyxy), dim=1)
                    metric = compute_metric(output[:, :6], target, iou_v)

            elif cls.shape[0] > 0: # If no objects detected but ground truth exists
                pass 

            # cls.squeeze(-1) always returns 1D or empty 1D tensor
            metrics.append((metric, conf_tensor_to_add, pred_cls_tensor_to_add, cls.squeeze(-1)))

        except Exception as e:
            print(f"Error processing image index {img_idx}: {e}")
            print("Image Path: ", eval_dataset.filenames[img_idx])
            # Skip this image and continue upon error (Optional)
            continue


    # Final mAP Calculation
    if not metrics:
         print("No results to evaluate")
         return

    try:
        # Debugging Start
        metrics_to_cat = list(zip(*metrics))
        metrics_processed_for_ap = []
        element_names = ["tp (metric)", "conf", "pred_cls", "target_cls"] # Define element names

        # Existing Logic (Executed if dimensions match)
        metrics_processed_for_ap = [torch.cat(x, dim=0).cpu().numpy() for x in metrics_to_cat]

        if len(metrics_processed_for_ap) > 0 and metrics_processed_for_ap[0].any(): # metrics[0] is tp_all
            print("\nCalculating Final Performance: ")
            # compute_ap function takes 4 numpy arrays as arguments (tp, conf, pred_cls, target_cls)
            tp, fp, m_pre, m_rec, map50, mean_ap = compute_ap(*metrics_processed_for_ap, plot=True, names=CLASS_NAMES)

            # Print Results
            print(('%10s' + '%10.3g' * 4) % ('', m_pre, m_rec, map50, mean_ap))
        else:
            print("Cannot calculate mAP as there are no valid detection results")

    except Exception as e:
        print(f"Error calculating final mAP: {e}")
        print("Check metrics list content or compute_ap function inputs")


if __name__ == "__main__":
    evaluate_tflite()