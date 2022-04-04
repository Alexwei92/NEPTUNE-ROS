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

MODEL_WEIGHT_PATH = 'affordance_model.pt'
ONNX_FILE_PATH = 'affordance_net.onnx'
TRT_PATH = 'affordance_net.trt'

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
    
    img = cv2.imread('sample_image.jpg', cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img_tensor = my_transform(img).unsqueeze(0)
    img_np = np.array(img_tensor)

    # Pytorch load time
    load_pytorch = timer('Load Pytorch Model')
    pytorch_model = AffordanceNet().eval().cuda()
    model_weight = torch.load('affordance_model.pt')
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
    with torch.no_grad():
        for i in range(10):
            tic = time.perf_counter()
            pytorch_outputs = pytorch_model(img_tensor.to(torch.device("cuda:0")))
            print(1/(time.perf_counter() - tic))
    print("Result: ", pytorch_outputs.cpu().squeeze(0).numpy())
    infer_pytorch.end()
    
    # ONNX infer time
    infer_onnx = timer('Run ONNX Infer')
    for i in range(10):
        tic = time.perf_counter()
        onnx_outputs = ort_session.run(None, {'input': img_np})[0]
        print(1/(time.perf_counter() - tic))
    print("Result: ", onnx_outputs[0])
    infer_onnx.end()

    # TRT infer time
    infer_trt = timer('Run TRT infer')
    with engine.create_execution_context() as context:
        for i in range(10):
            tic = time.perf_counter()
            trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            print(1/(time.perf_counter() - tic))
    print("Result: ", trt_outputs[0])
    infer_trt.end()
