import os
import sys
import torch
import tensorrt as trt
import argparse

from log_utils import timer, logger

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
from models import EndToEnd_NoDropout

argParser = argparse.ArgumentParser()
argParser.add_argument("-o", "--option", type=int, default=3, help="model option")
argParser.add_argument("-extra", "--extra", default=False, action="store_true", help="enable state extra")

args = argParser.parse_args()
enable_extra = args.extra
MODEL_OPTION = args.option
print(f"ENABLE_EXTRA: {enable_extra}")
print(f"MODEL_OPTION: {MODEL_OPTION}")

if MODEL_OPTION == 1: # Affordance Net
    MODEL_WEIGHT_PATH = os.path.join(model_weight_dir, 'affordance/affordance_model.pt')
    ONNX_FILE_PATH = os.path.join(model_weight_dir, 'affordance/affordance_net.onnx')
    TRT_PATH = os.path.join(model_weight_dir, 'affordance/affordance_net.trt')

if MODEL_OPTION == 2: # VAE 
    MODEL_WEIGHT_PATH = os.path.join(model_weight_dir, 'vae/vanilla_vae_model_z_1000.pt')
    ONNX_FILE_PATH = os.path.join(model_weight_dir, 'vae/vanilla_vae_model_z_1000.onnx')
    TRT_PATH = os.path.join(model_weight_dir, 'vae/vanilla_vae_model_z_1000.trt')

if MODEL_OPTION == 3: # VAELatentCtrl
    if enable_extra:
        MODEL_WEIGHT_PATH = os.path.join(model_weight_dir, 'vae/combined_vae_latent_ctrl_z_1000_with_extra.pt')
        ONNX_FILE_PATH = os.path.join(model_weight_dir, 'vae/combined_vae_latent_ctrl_z_1000_with_extra.onnx')
        TRT_PATH = os.path.join(model_weight_dir, 'vae/combined_vae_latent_ctrl_z_1000_with_extra.trt')
    else:
        MODEL_WEIGHT_PATH = os.path.join(model_weight_dir, 'vae/combined_vae_latent_ctrl_z_1000_no_extra.pt')
        ONNX_FILE_PATH = os.path.join(model_weight_dir, 'vae/combined_vae_latent_ctrl_z_1000_no_extra.onnx')
        TRT_PATH = os.path.join(model_weight_dir, 'vae/combined_vae_latent_ctrl_z_1000_no_extra.trt')

if MODEL_OPTION == 4: # EndToEnd
    MODEL_WEIGHT_PATH = os.path.join(model_weight_dir, 'endToend/end_to_end_model.pt')
    ONNX_FILE_PATH = os.path.join(model_weight_dir, 'endToend/end_to_end_model.onnx')
    TRT_PATH = os.path.join(model_weight_dir, 'endToend/end_to_end_model.trt')


def build_engine(onnx_path):
    with trt.Builder(trt_logger) as builder, builder.create_network(1) as network, trt.OnnxParser(network, trt_logger) as parser:
        config = builder.create_builder_config()
        # config.max_workspace_size = (1 << 30)
        # config.set_flag(trt.BuilderFlag.FP16) # if use FP_16 precision
        parser.parse_from_file(onnx_path)        
        serialized_engine = builder.build_serialized_network(network, config)
        return serialized_engine

def save_engine(serialized_engine, engine_path):
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)


if __name__ == '__main__':

    trt_logger = trt.Logger(trt.Logger.ERROR)
    trt_runtime = trt.Runtime(trt_logger)

    # Load Pytorch model and weight
    load_pytorch = timer("Loading Pytorch model")

    if MODEL_OPTION == 1:
        model = AffordanceNet_Resnet18(
            name='affordance',
            input_dim=256,
            output_dim=2,
            n_image=4).eval().cuda()

        sample_input = torch.ones((1, 3*4, 256, 256)).cuda()

    if MODEL_OPTION == 2:
        model = VanillaVAE(
            name='vanilla_vae',
            input_dim=128,
            in_channels=3,
            z_dim=1000).eval().cuda()

        sample_input = torch.ones((1, 3, 128, 128)).cuda()

    if MODEL_OPTION == 3:
        model = VAELatentCtrl(
            name='vae_latent_ctrl',
            input_dim=128,
            in_channels=3,
            z_dim=1000, 
            extra_dim=6).eval().cuda()

        sample_input = (torch.ones((1, 3, 128, 128)).cuda(), torch.zeros((1,6)).cuda())

    if MODEL_OPTION == 4:
        model = EndToEnd_NoDropout(
            name='end_to_end',
            in_channels=3,
            input_dim=128,
            extra_dim=6,
        ).eval().cuda()

        sample_input = (torch.ones((1, 3, 128, 128)).cuda(), torch.zeros((1,6)).cuda())

    model_weight = torch.load(MODEL_WEIGHT_PATH)
    model.load_state_dict(model_weight)
    load_pytorch.end()

    # Pytorch to onnx
    pytorch_to_onnx = timer("Convert Pytorch to ONNX file")
    torch.onnx.export(model, sample_input, ONNX_FILE_PATH, input_names=['input'],
                        output_names=['output'], export_params=True, training=torch.onnx.TrainingMode.EVAL)
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