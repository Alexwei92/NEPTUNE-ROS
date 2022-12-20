import os
import sys
import cv2
import torch
import numpy as np
from torchvision import transforms
import onnxruntime as ort
import tensorrt as trt
import time

from log_utils import timer, logger
import common

#################
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir)
model_weight_dir = os.path.join(root_dir, 'model_weight')
model_dir = os.path.abspath(os.path.join(root_dir, "src/network_training/scripts"))
sys.path.insert(0, model_dir)
#################

from models import AffordanceNet_Resnet18
from models import VanillaVAE
from models import VAELatentCtrl

MODEL_OPTION = 3

if MODEL_OPTION == 1: # Affordance Net
    MODEL_WEIGHT_PATH = os.path.join(model_weight_dir, 'affordance/affordance_model.pt')
    ONNX_FILE_PATH = os.path.join(model_weight_dir, 'affordance/affordance_net.onnx')
    TRT_PATH = os.path.join(model_weight_dir, 'affordance/affordance_net.trt')

if MODEL_OPTION == 2: # VAE 
    MODEL_WEIGHT_PATH = os.path.join(model_weight_dir, 'vae/vanilla_vae_model_z_1000.pt')
    ONNX_FILE_PATH = os.path.join(model_weight_dir, 'vae/vanilla_vae_model_z_1000.onnx')
    TRT_PATH = os.path.join(model_weight_dir, 'vae/vanilla_vae_model_z_1000.trt')

if MODEL_OPTION == 3: # VAELatentCtrl
    MODEL_WEIGHT_PATH = os.path.join(model_weight_dir, 'vae/combined_vae_latent_ctrl_z_1000.pt')
    ONNX_FILE_PATH = os.path.join(model_weight_dir, 'vae/combined_vae_latent_ctrl_z_1000.onnx')
    TRT_PATH = os.path.join(model_weight_dir, 'vae/combined_vae_latent_ctrl_z_1000.trt')


def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        serialized_engine  = f.read()
    engine = trt_runtime.deserialize_cuda_engine(serialized_engine)
    return engine


if __name__ == '__main__':

    trt_logger = trt.Logger(trt.Logger.ERROR)
    trt_runtime = trt.Runtime(trt_logger)

    my_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])
    
    img = cv2.imread('sample_image.png', cv2.IMREAD_UNCHANGED)

    if MODEL_OPTION == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img_tensor = my_transform(img)
        img_tensor_list = []
        for i in range(4):
            img_tensor_list.append(img_tensor)
        img_tensor = torch.cat(tuple(img for img in img_tensor_list), dim=0)

        show_result = True

    if MODEL_OPTION == 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img_tensor = my_transform(img)

        show_result = False

    if MODEL_OPTION == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img_tensor = my_transform(img)

        state_extra = np.zeros(5, dtype=np.float32)
        state_extra_tensor = torch.from_numpy(state_extra).unsqueeze(0)
        state_extra = np.array(state_extra_tensor)

        show_result = True

    img_tensor = img_tensor.unsqueeze(0)
    img_np = np.array(img_tensor)

    # Pytorch load time
    load_pytorch = timer('Load Pytorch Model')
    if MODEL_OPTION == 1:
        pytorch_model = AffordanceNet_Resnet18(
            name='affordance',
            input_dim=256,
            output_dim=2,
            n_image=4).eval().cuda()

    if MODEL_OPTION == 2:
        pytorch_model = VanillaVAE(
            name='vanilla_vae',
            input_dim=128,
            in_channels=3,
            z_dim=1000).eval().cuda()

    if MODEL_OPTION == 3:
        pytorch_model = VAELatentCtrl(
            name='vae_latent_ctrl',
            input_dim=128,
            in_channels=3,
            z_dim=1000, 
            extra_dim=5).eval().cuda()

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
    if MODEL_OPTION == 3:
        inputs[0].host = img_np
        inputs[1].host = state_extra
    else:
        inputs[0].host = img_np

    # Pytorch infer time
    infer_pytorch = timer('Run Pytorch Infer')
    print("\n")
    with torch.no_grad():
        for i in range(2): # warm start
            if MODEL_OPTION == 3:
                pytorch_outputs = pytorch_model(img_tensor.to(torch.device("cuda:0")), state_extra_tensor.to(torch.device("cuda:0")))
            else:
                pytorch_outputs = pytorch_model(img_tensor.to(torch.device("cuda:0")))
        for i in range(10):
            tic = time.perf_counter()
            if MODEL_OPTION == 3:
                pytorch_outputs = pytorch_model(img_tensor.to(torch.device("cuda:0")), state_extra_tensor.to(torch.device("cuda:0"))) 
            else:
                pytorch_outputs = pytorch_model(img_tensor.to(torch.device("cuda:0"))) 
            print(1/(time.perf_counter() - tic), "Hz")
    if show_result: print("Result: ", pytorch_outputs.cpu().squeeze(0).numpy())
    infer_pytorch.end()
    
    # ONNX infer time
    infer_onnx = timer('Run ONNX Infer')
    print("\n")
    for i in range(2): # warm start
        if MODEL_OPTION == 3:
            onnx_input = {ort_session.get_inputs()[0].name: img_np, ort_session.get_inputs()[1].name: state_extra}
        else:
            onnx_input = {ort_session.get_inputs()[0].name: img_np}
        ort_session.run(None, onnx_input)[0]
    for i in range(10):
        tic = time.perf_counter()
        onnx_outputs = ort_session.run(None, onnx_input)[0]
        print(1/(time.perf_counter() - tic), "Hz")
    if show_result: print("Result: ", onnx_outputs[0])
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
    if show_result: print("Result: ", trt_outputs[0])
    infer_trt.end()
