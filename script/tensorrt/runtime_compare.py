import os
import cv2
import torch
import numpy as np
from torchvision import transforms
import onnxruntime as ort
import tensorrt as trt
import time

from affordance_net import AffordanceNet
from log_utils import timer, logger
import common

trt_logger = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(trt_logger)

curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
root_dir = os.path.dirname(parent_dir)
model_dir = os.path.join(root_dir, 'model')

MODEL_WEIGHT_PATH = os.path.join(model_dir, 'affordance/affordance_model.pt')
ONNX_FILE_PATH = os.path.join(model_dir, 'affordance/affordance_net.onnx')
TRT_PATH = os.path.join(model_dir, 'affordance/affordance_net.trt')

def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        serialized_engine  = f.read()
    engine = trt_runtime.deserialize_cuda_engine(serialized_engine)
    return engine

if __name__ == '__main__':
    my_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])
    
    input_dim = 256
    output_dim = 2
    n_image = 4

    img = cv2.imread('sample_image.png', cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_dim, input_dim))
    img_tensor = my_transform(img)
    img_tensor_list = []
    for i in range(n_image):
        img_tensor_list.append(img_tensor)
    img_tensor = torch.cat(tuple(img for img in img_tensor_list), dim=0)
    img_tensor = img_tensor.unsqueeze(0)
    img_np = np.array(img_tensor)

    # Pytorch load time
    load_pytorch = timer('Load Pytorch Model')
    pytorch_model = AffordanceNet(
                input_dim=input_dim,
                output_dim=output_dim,
                n_image=n_image).eval().cuda()
    model_weight = torch.load(MODEL_WEIGHT_PATH)
    pytorch_model.load_state_dict(model_weight)
    load_pytorch.end()
    
    # ONNX load time
    load_onnx = timer('Load ONNX Model')
    ort_session = ort.InferenceSession(ONNX_FILE_PATH, providers=['CUDAExecutionProvider'])
    load_onnx.end()

    # TensorRT load time
    load_trt = timer('Load TRT Model')
    engine = load_engine(trt_runtime, TRT_PATH)
    load_trt.end()
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    inputs[0].host = img_np

    # Pytorch infer time
    infer_pytorch = timer('Run Pytorch Infer')
    print("\n")
    with torch.no_grad():
        for i in range(2): # warm start
            pytorch_outputs = pytorch_model(img_tensor.to(torch.device("cuda:0")))
        for i in range(10):
            tic = time.perf_counter()
            pytorch_outputs = pytorch_model(img_tensor.to(torch.device("cuda:0")))
            print(1/(time.perf_counter() - tic), "Hz")
    print("Result: ", pytorch_outputs.cpu().squeeze(0).numpy())
    infer_pytorch.end()
    
    # ONNX infer time
    infer_onnx = timer('Run ONNX Infer')
    print("\n")
    for i in range(2): # warm start
        ort_session.run(None, {'input': img_np})[0]
    for i in range(10):
        tic = time.perf_counter()
        onnx_outputs = ort_session.run(None, {'input': img_np})[0]
        print(1/(time.perf_counter() - tic), "Hz")
    print("Result: ", onnx_outputs[0])
    infer_onnx.end()

    # TRT infer time
    infer_trt = timer('Run TRT infer')
    print("\n")
    with engine.create_execution_context() as context:
        for i in range(2): # warm start
            common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        for i in range(10):
            tic = time.perf_counter()
            trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            print(1/(time.perf_counter() - tic), "Hz")
    print("Result: ", trt_outputs[0])
    infer_trt.end()
