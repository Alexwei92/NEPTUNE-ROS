import torch
import onnx
from torch2trt import torch2trt
from torchvision.models.alexnet import alexnet
from affordance import AffordanceNet
import time

# model = alexnet(pretrained=True).eval().cuda()
model = AffordanceNet().eval().cuda()

x = torch.ones((1, 3, 256, 256)).cuda()
ONNX_FILE_PATH = 'affordance.onnx'
torch.onnx.export(model, x, ONNX_FILE_PATH, input_names=['input'],
	                output_names=['output'], export_params=True)

onnx_model = onnx.load(ONNX_FILE_PATH)
onnx.checker.check_model(onnx_model)

# create some regular pytorch model...
# model = AffordanceNet().eval().cuda()

# create example data
# x = torch.ones((1, 3, 256, 256)).cuda()

# convert to TensorRT feeding sample data as input
# model_trt = torch2trt(model, [x])

# tic = time.perf_counter()
# y = model(x)
# tac = time.perf_counter()
# print("Elapsed Time: {:.2f}".format(tac-tic))

# tic = time.perf_counter()
# y_trt = model_trt(x)
# tac = time.perf_counter()
# print("Elapsed Time: {:.2f}".format(tac-tic))

# check the output against PyTorch
# print(torch.max(torch.abs(y - y_trt)))

# torch.save(model_trt.state_dict(), 'alexnet_trt.pth')

# from torch2trt import TRTModule

# model_trt = TRTModule()

# model_trt.load_state_dict(torch.load('alexnet_trt.pth'))