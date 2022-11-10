import torch
from affordance_net import AffordanceNet
import tensorrt as trt

from log_utils import timer, logger

trt_logger = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(trt_logger)

MODEL_WEIGHT_PATH = 'affordance_model.pt'
ONNX_FILE_PATH = 'affordance_net.onnx'
TRT_PATH = 'affordance_net.trt'

def build_engine(onnx_path):
    with trt.Builder(trt_logger) as builder, builder.create_network(1) as network, trt.OnnxParser(network, trt_logger) as parser:
        config = builder.create_builder_config()
        config.max_workspace_size = (1 << 30)
        # config.set_flag(trt.BuilderFlag.FP16) # if use FP_16 precision
        parser.parse_from_file(onnx_path)        
        serialized_engine = builder.build_serialized_network(network, config)
        return serialized_engine

def save_engine(serialized_engine, engine_path):
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)


if __name__ == '__main__':

    # Load Pytorch model and weight
    load_pytorch = timer("Loading Pytorch model")
    input_dim = 256
    output_dim = 2
    n_image = 4
    model = AffordanceNet(
            input_dim=input_dim,
            output_dim=output_dim,
            n_image=n_image).eval().cuda()
    model_weight = torch.load(MODEL_WEIGHT_PATH)
    model.load_state_dict(model_weight)
    load_pytorch.end()

    # Pytorch to onnx
    pytorch_to_onnx = timer("Convert Pytorch to ONNX file")
    sample_input = torch.ones((1, 3*n_image, input_dim, input_dim)).cuda()
    torch.onnx.export(model, sample_input, ONNX_FILE_PATH, input_names=['input'],
                        output_names=['output'], export_params=True)
    pytorch_to_onnx.end()

    # check the onnx model
    import onnx
    onnx_model = onnx.load(ONNX_FILE_PATH)
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))

    build_trt = timer('Parser ONNX & Build TensorRT Engine')
    engine = build_engine(ONNX_FILE_PATH)
    build_trt.end()

    save_trt = timer('Save TensorRT Engine')
    save_engine(engine, TRT_PATH)
    save_trt.end()

    print("Successfully generated the .trt file!")