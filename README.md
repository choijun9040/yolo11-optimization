# YOLO11 Optimization

**NPU-Aware Optimization of YOLO11 for ARM Ethos-N**

ARM Ethos-N을 위한 YOLO11의 NPU 인지형 최적화 파이프라인 구축

![Image](https://github.com/user-attachments/assets/0ff3ce14-3876-48b8-af9a-13a8f55730c4)

### Installation

```
conda create -n YOLO python=3.10.10
conda activate YOLO
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install opencv-python
pip install PyYAML
pip install tqdm
```

### Dataset structure

    ├── MyFirstProject 
        ├── images
            ├── train
                ├── 1111.jpg
                ├── 2222.jpg
            ├── val2017
                ├── 1111.jpg
                ├── 2222.jpg
        ├── labels
            ├── train
                ├── 1111.txt
                ├── 2222.txt
            ├── val
                ├── 1111.txt
                ├── 2222.txt
        ├── train.txt           # Path to train images  e.g, images/train/1111.jpg
        ├── val.txt             # Path to val images    e.g, images/val/1111.jpg   
        ├── test.txt            # Path to test images   e.g, images/test/1111.jpg

### Workflow
---

#### Training
* Configure your dataset path in `main.py` for training
* Run `bash main.sh $ --train` for training, `$` is number of GPUs

#### Testing
* Configure your dataset path in `main.py` for testing
* Run `python main.py --test` for testing

#### Quantization-Aware Training (QAT)
* Configure your dataset path in `main.py` for training
* Run `python main.py --train --qat` for quantization-aware training

#### Testing (QAT)
* Configure your dataset path in `main.py` for testing
* Run `python main.py --test --qat` for testing the QAT model

#### PyTorch to ONNX
* Run `python pt2onnx.py` for exporting the PyTorch model to ONXN format

#### ONNX to TFLite
* Run `python onnx2tflite.py` for converting the ONNX model to INT8 TFLite format

#### Testing (TFLite)
* Run `python test_tflite.py` for testing the INT8 quantized TFLite model

#### Model Compilation by TVM tool
* Compile the TFLite model to generate a `.tar` model archive for deployment on the D5 board

#### JPG to NPZ
* Run `jpg2npz.py` for converting input JPG images to NPZ format for deployment on the D5 board

#### RTVM inference
* Deploy the `.tar` model archive and `.npz` inputs to the D5 board to run inference and save the results as an output `.npz` file.

#### Testing (Output of RTVM inference)
* Run `eval_output_npz.py` for testing the accuracy of the output `.npz` files

#### Post-processing and Visualization
* Run `postprocess.py` for post-processing the output `.npz` file and visualizing the inference results

### Results
#### mAP@50
| Model     |  mAP@50   |Params size (MB)|                                                                              Download                 |
|:-------:  |:-------:  |:-------------: |:--------------------------------------------------------------------------------------:               |
|  PyTorch  |   94.9%   |      12.0      |[Model](https://github.com/choijun9040/yolo11-optimization/releases/download/v0.0.1/fp32_best.pt)      |
|PyTorch_QAT|   95.7%   |      12.0      |[Model](https://github.com/choijun9040/yolo11-optimization/releases/download/v0.0.1/fp32_qat_best.pt)  |
|   ONNX    |     -     |      11.0      |[Model](https://github.com/choijun9040/yolo11-optimization/releases/download/v0.0.1/onnx_best.onnx)    |
|  TFLite   |   94.8%   |      3.0       |[Model](https://github.com/choijun9040/yolo11-optimization/releases/download/v0.0.1/int8_best.tflite)  |

#### Precision-Recall curve
<img width="270" height="180" alt="Image" src="https://github.com/user-attachments/assets/ef29d97a-0a51-4348-a3d4-85f9e16b12ff" />

<img width="270" height="180" alt="Image" src="https://github.com/user-attachments/assets/a8f4bc12-4f84-4203-87eb-b01caa7381c7" />

<img width="270" height="180" alt="Image" src="https://github.com/user-attachments/assets/ea3231a4-f8de-4e2b-86f9-0504e1681a62" />

#### mAP@50(Output of RTVM inference)
<img width="600" height="440" alt="Image" src="https://github.com/user-attachments/assets/e97b91a2-6aeb-424c-8543-6d87449362dd" />

#### Detection Visualization Reulst
![Image](https://github.com/user-attachments/assets/58922870-f658-434e-8d77-3c5c1e29f3f8)


### Reference
* https://github.com/ultralytics/ultralytics
* https://github.com/jahongir7174/YOLOv11-pt
