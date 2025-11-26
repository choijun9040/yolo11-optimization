import onnx2tf
import yaml
import numpy as np
import torch
import os
from utils.dataset import Dataset

ONNX_PATH = "best.onnx"
OUTPUT_TFLITE_PATH = "best"
INPUT_SIZE = 640
DATA_DIR = '../Dataset/MyFirstProject'
VAL_FILE = 'val.txt'                   # List of images to use
NUM_CALIBRATION_IMAGES = 200            # Number of images to use for calibration
CALIBRATION_DATA_PATH = "calibration_data.npy" # Name of the Numpy file to generate

# Load YAML parameters
try:
    with open('utils/args.yaml', errors='ignore') as f:
        PARAMS = yaml.safe_load(f)
except Exception as e:
    print(f"Error: Failed to load 'utils/args.yaml': {e}")
    exit()

# Function to generate representative dataset .npy file
def prepare_calibration_data():
    """
    Prepares NumPy data for TFLite PTQ calibration and saves it to a file
    """
    print(f"Starting image load from '{VAL_FILE}' to generate representative dataset")
    
    # Load list of validation image files
    filenames = []
    image_folder = os.path.join(DATA_DIR, 'images', 'val')
    try:
        with open(f'{DATA_DIR}/{VAL_FILE}') as f:
            for line in f.readlines():
                filename = os.path.basename(line.strip())
                if not filename: continue
                full_img_path = os.path.join(image_folder, filename)
                if os.path.exists(full_img_path):
                     filenames.append(full_img_path)
    except Exception as e:
        print(f"Error: Failed to load calibration image list: {e}")
        return None

    if not filenames:
        print(f"Error: No calibration images found in '{DATA_DIR}/{VAL_FILE}'")
        return None
        
    # Create Dataset object
    dataset = Dataset(filenames, INPUT_SIZE, PARAMS, augment=False)
    
    num_to_load = min(NUM_CALIBRATION_IMAGES, len(dataset))
    print(f"Loading a total of {num_to_load} images for calibration")
    
    all_images = []
    for i in range(num_to_load):
        try:
            # Load tensor: (3, H, W), FP32, [0, 255]
            sample_tensor, _, _, _ = dataset[i]
            
            # Normalize to [0.0, 1.0] range to match model input (QuantStub)
            img_fp32_1range = sample_tensor.numpy().astype(np.float32) / 255.0
            
            # Transpose to NHWC: (3, 640, 640) -> (640, 640, 3)
            img_nhwc = np.transpose(img_fp32_1range, (1, 2, 0))

            # Add batch dimension (1) to NumPy array
            # (640, 640, 3) -> (1, 640, 640, 3)
            img_nhwc_batch = np.expand_dims(img_nhwc, axis=0)
            all_images.append(img_nhwc_batch)
            
        except Exception as e:
            print(f"Warning: Failed to load data at index {i}: {e}")

    # Concatenate all images into a single NumPy array (axis=0)
    # [(1, 3, H, W), (1, 3, H, W), ...] -> (num_to_load, 3, H, W)
    if not all_images:
        print("Error: No calibration images loaded")
        return None
        
    calibration_data_array = np.concatenate(all_images, axis=0)
    
    # Save as NumPy file (.npy)
    try:
        np.save(CALIBRATION_DATA_PATH, calibration_data_array)
        print(f"Representative dataset saved to '{CALIBRATION_DATA_PATH}'. (Shape: {calibration_data_array.shape})")
        return CALIBRATION_DATA_PATH
    except Exception as e:
        print(f"Error: Failed to save NumPy file: {e}")
        return None

print(f"Starting '{ONNX_PATH}' -> '{OUTPUT_TFLITE_PATH}' INT8 conversion (including PTQ calibration)")
print(f"(Fixing input size to {INPUT_SIZE}x{INPUT_SIZE})")

# Prepare calibration .npy file
calib_data_file = prepare_calibration_data()

if calib_data_file is None:
    print("Error: Failed to prepare calibration data file. Aborting conversion")
    exit()

# Create argument list for onnx2tf
# Format: [[INPUT_NAME, NUMPY_FILE_PATH, MEAN_FLOAT, STD_FLOAT]]
# Pass actual floats 0.0 and 1.0, not string '0.0'
calibration_path_list = [
    [
        'images',                 # 1. INPUT_NAME
        calib_data_file,          # 2. NUMPY_FILE_PATH
        0.0,                      # 3. MEAN (Float)
        1.0                       # 4. STD (Float)
    ]
]
print(f"Passing the following arguments to onnx2tf: {calibration_path_list}")

try:
    # Call onnx2tf convert function with correct argument names
    onnx2tf.convert(
        input_onnx_file_path=ONNX_PATH,
        output_folder_path=OUTPUT_TFLITE_PATH,
        
        # Enable INT8 quantization (Use QAT model info)
        output_integer_quantized_tflite=True,
        
        # Pass correct argument name and file path list (including mean/std)
        custom_input_op_name_np_data_path=calibration_path_list,
        
        # Fix input shape to resolve dynamic axis issues
        overwrite_input_shape=[f"images:1,3,{INPUT_SIZE},{INPUT_SIZE}"],
        
        # (Optional) To see more detailed conversion logs
        # verbose=True 
    )

    print(f"\nConversion task completed")
    print(f"Please check the .tflite file in the '{OUTPUT_TFLITE_PATH}' folder")

except Exception as e:
    print(f"\nonnx2tf conversion failed: {e}")
    print("Please check the onnx2tf library version or verify the ONNX file")