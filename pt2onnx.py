import torch
import yaml
from nets import nn
from torch.quantization import MinMaxObserver, QConfig, QuantStub, DeQuantStub

# Settings
WEIGHTS_PATH = './weights_qat/best.pt' # Path to weights generated from QAT retraining
ONNX_OUTPUT_PATH = "best.onnx"
INPUT_SIZE = 640

print("Loading configuration file (utils/args.yaml)")
with open('utils/args.yaml', errors='ignore') as f:
    params = yaml.safe_load(f)
num_classes = len(params['names'])

# Create an empty FP32 model structure first
print("Creating YOLO v11-n model structure")
model = nn.yolo_v11_n(num_classes)

# Apply specific QAT configurations based on the layer type
print("Converting model structure to QAT state (applying custom qconfig)")
model.train() # It is safer to apply prepare_qat in train mode

# per_channel_qconfig = QConfig(
#     activation=MinMaxObserver.with_args(dtype=torch.quint8),
#     weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
# )
per_tensor_qconfig = QConfig(
    activation=MinMaxObserver.with_args(dtype=torch.quint8),
    weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
)

for name, module in model.named_modules():
    # if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
    #     module.qconfig = per_channel_qconfig
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.ConvTranspose2d, QuantStub, DeQuantStub)):
        module.qconfig = per_tensor_qconfig

# Apply the final QAT configuration to the model
torch.quantization.prepare_qat(model, inplace=True)

# Load the weights (state_dict) from the QAT retraining
print(f"Loading QAT-retrained weights '{WEIGHTS_PATH}'")
state_dict = torch.load(WEIGHTS_PATH, map_location='cpu')['model']
model.load_state_dict(state_dict)

# Switch the model to evaluation mode
# Calling eval() switches QAT modules to inference mode
model.eval()
print("Model switched to evaluation mode")


print("Moving model and input to CPU for ONNX export")
model_cpu = model.to('cpu')
dummy_input_cpu = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)

# Export the 'fake quantized' model to ONNX
print(f"Exporting to ONNX file '{ONNX_OUTPUT_PATH}' containing quantization information")
torch.onnx.export(
    model_cpu,
    dummy_input_cpu,
    ONNX_OUTPUT_PATH,
    export_params=True,
    opset_version=13,
    input_names=['images'],
    # Specify output names matching the model's actual output count
    output_names=['output0', 'output1', 'output2'],
    # It is safer to specify dynamic axes for multiple outputs as shown below
    dynamic_axes={
        'images': {0: 'batch_size'},
        'output0': {0: 'batch_size'},
        'output1': {0: 'batch_size'},
        'output2': {0: 'batch_size'},
    }
)

print(f"'{ONNX_OUTPUT_PATH}' file has been generated")
print("Now you can convert this ONNX file to TFLite and proceed with TVM compilation")