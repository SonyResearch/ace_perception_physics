# Confidential, Copyright 2024, Sony AI, All rights reserved.
"""
Example to load onnx model into trt engine
Confidential, Copyright 2024, Sony AI, All rights reserved
"""

import os
import time
import numpy as np
import cv2
from ament_index_python.packages import get_package_share_directory
import cuda_common.python_module as cuda_common

# an example to load an onnx model and build a TensorRT engine
base_path = get_package_share_directory("cuda_common")
base_path = os.path.join(base_path, "test")
model_path = os.path.join(base_path, "yolov8n.onnx")
image_path = os.path.join(base_path, "test.jpg")

options = cuda_common.TRTEngineOptions()
options.precision = cuda_common.EnginePrecision.FP16
options.opt_batch_size = 1
options.max_batch_size = 1
options.input_format = cuda_common.EngineInputFormat.BCHW
engine = cuda_common.TRTEngine(options)
engine.build(model_path)
engine.load_network()

print(engine.get_input_dims())
print(engine.get_output_dims())

print("Index of `images` input:", engine.get_input_index("images"))

input_dims = engine.get_input_dims()[engine.get_input_index("images")]


image = cv2.imread(image_path)
image = cv2.resize(image, (input_dims[1], input_dims[2]))

N = 200
for i in range(N):
    ret = engine.inference_single([image])
t1 = time.time()
N = 1000
for i in range(N):
    ret = engine.inference_single([image])

average = (time.time() - t1) / N
np.printoptions(suppress=True, precision=3)
print(f"Average time to run single inference: { np.array(average*1e3)}ms")
# or if the model support batch inference
# ret, output=engine.inference_batch([image])

print("Inference success: ", ret)
output = engine.get_output()
print("Inference output size: ", np.array(output).shape)

yolo_out = np.array(output).reshape(engine.get_output_dims()[0])

# you can run here yolo extraction..
