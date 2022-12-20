import torch
import os
import cv2
import numpy as np
from torchvision import transforms
import tensorrt as trt

from models import VanillaVAE, LatentCtrl, VAELatentCtrl
from utils import tensorrt_utils

########## GLOBAL VARIABLES ############
curr_dir        = os.path.dirname(os.path.abspath(__file__))
parent_dir      = os.path.dirname(curr_dir)
config_dir      = os.path.join(parent_dir, 'configs')

vae_model_config = {
    'name': 'vanilla_vae',
    'in_channels': 3,
    'z_dim': 1000,
    'input_dim': 128,
}

latent_ctrl_model_config = {
    'name': 'latent_ctrl',
    'z_dim': vae_model_config['z_dim'],
    'extra_dim': 5,
}

vae_latent_ctrl_model_config = {
    'name': 'vae_latent_ctrl',
    'input_dim': vae_model_config['input_dim'],
    'in_channels': vae_model_config['in_channels'],
    'z_dim': vae_model_config['z_dim'],
    'extra_dim': latent_ctrl_model_config['extra_dim'],
}
########################################

class VAELatentController_Full():
    """
    VAE-Based Controller with full size VAE and LatentCtrl
    """
    def __init__(self, **kwargs): 
        self.configure(**kwargs)
        self.load_model(**kwargs)
 
    def configure(self, **kwargs):
        """
        Configure
        """
        self.device             = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform_composed = transforms.Compose([ 
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5)),
                                ]) 
        
    def load_model(self, vae_model_weight_path, latent_model_weight_path, **kwargs):
        """
        Load Model
        """
        self.load_VAE_model(vae_model_weight_path)
        self.load_latent_model(latent_model_weight_path)

    def load_VAE_model(self, model_weight_path):
        """
        Load VAE model
        """
        model_weight = torch.load(model_weight_path)
        self.VAE_model = VanillaVAE(**vae_model_config).to(self.device)
        self.VAE_model.load_state_dict(model_weight)
        self.z_dim = self.VAE_model.z_dim
        self.image_resize = [self.VAE_model.input_dim, self.VAE_model.input_dim]

    def load_latent_model(self, model_weight_path):
        """
        Load Latent FC model
        """
        model_weight = torch.load(model_weight_path)
        self.Latent_model = LatentCtrl(**latent_ctrl_model_config).to(self.device)
        self.Latent_model.load_state_dict(model_weight)

    def predict(self, image_color, is_bgr=True, state_extra=None):
        """
        Predict the action
        """
        image_np = image_color.copy() # hard copy
        if is_bgr:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_np = cv2.resize(image_np, (self.image_resize[1], self.image_resize[0]))
        image_tensor = self.transform_composed(image_np)
        
        self.VAE_model.eval()
        self.Latent_model.eval()
        with torch.no_grad():
            z = self.VAE_model.get_latent(image_tensor.unsqueeze(0).to(self.device), with_logvar=False)
            if state_extra is not None:
                state_extra = state_extra.astype(np.float32)
                y_pred = self.Latent_model(z, torch.from_numpy(state_extra).unsqueeze(0).to(self.device))
            else:
                y_pred = self.Latent_model(z)
                
            y_pred = y_pred.cpu().item()

        return y_pred

    def reconstruct_image(self, image_color, is_bgr=True):
        """
        Reconstruct the images from VAE for visualization
        """
        image_np = image_color.copy()
        if is_bgr:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_np = cv2.resize(image_np, (self.image_resize[1], self.image_resize[0]))
        image_tensor = self.transform_composed(image_np)
        
        self.VAE_model.eval()
        with torch.no_grad():
            reconst_image = self.VAE_model(image_tensor.unsqueeze(0).to(self.device))
            
        image_pred = reconst_image[0].cpu().squeeze(0).numpy()
        image_raw = reconst_image[1].cpu().squeeze(0).numpy()
        
        image_pred = ((image_pred + 1.0) / 2.0 * 255.0).astype(np.uint8)
        image_raw = ((image_raw + 1.0) / 2.0 * 255.0).astype(np.uint8)
        
        return image_raw.transpose(1,2,0), image_pred.transpose(1,2,0) # raw, reconstructed


class VAELatentController():
    """
    VAE-Based Controller with reduced size VAE and LatentCtrl
    """
    def __init__(self, **kwargs): 
        self.configure(**kwargs)
        self.load_model(**kwargs)
 
    def configure(self, **kwargs):
        """
        Configure
        """
        self.input_dim          = vae_latent_ctrl_model_config['input_dim']
        self.device             = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform_composed = transforms.Compose([ 
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5)),
                                ]) 
        
    def load_model(self, model_weight_path, **kwargs):
        """
        Load Model
        """
        self.model = VAELatentCtrl(**vae_latent_ctrl_model_config).to(self.device)
        model_weight = torch.load(model_weight_path)
        self.model.load_state_dict(model_weight)

    def predict(self, image_color, is_bgr=True, state_extra=None):
        """
        Predict the action
        """
        image_np = image_color.copy() # hard copy
        if is_bgr:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_np = cv2.resize(image_np, (self.model.input_dim, self.model.input_dim))
        image_tensor = self.transform_composed(image_np)
        
        self.model.eval()
        with torch.no_grad():
            if state_extra is not None:
                state_extra = state_extra.astype(np.float32)
                y_pred = self.model(image_tensor.unsqueeze(0).to(self.device), torch.from_numpy(state_extra).unsqueeze(0).to(self.device))
            else:
                y_pred = self.model(image_tensor.unsqueeze(0).to(self.device))
                
            y_pred = y_pred.cpu().item()
        
        return y_pred


class VAELatentController_TRT():
    """
    VAE-Based Controller with reduced size VAE and LatentCtrl (run with TensorRT)
    """
    def __init__(self, **kwargs): 
        self.configure(**kwargs)
        self.load_engine(**kwargs)
 
    def configure(self, **kwargs):
        """
        Configure
        """
        self.input_dim          = vae_latent_ctrl_model_config['input_dim']
        self.transform_composed = transforms.Compose([ 
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5)),
                                ]) 

    def load_engine(self, tensorrt_engine_path, **kwargs):
        """
        Load TensorRT Engine
        """
        tensorrt_logger = trt.Logger(trt.Logger.ERROR)
        tensorrt_runtime = trt.Runtime(tensorrt_logger)
        with open(tensorrt_engine_path, 'rb') as f:
            serialized_engine  = f.read()
        self.engine = tensorrt_runtime.deserialize_cuda_engine(serialized_engine)

        self.inputs, self.outputs, self.bindings, self.stream = tensorrt_utils.allocate_buffers(self.engine)

    def predict(self, image_color, is_bgr=True, state_extra=None):
        """
        Predict the action
        """
        image_np = image_color.copy() # hard copy
        if is_bgr:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_np = cv2.resize(image_np, (self.input_dim, self.input_dim))
        image_tensor = self.transform_composed(image_np)

        with self.engine.create_execution_context() as context:
            self.inputs[0].host = np.array(image_tensor, dtype=np.float32)
            self.inputs[1].host = np.array(state_extra, dtype=np.float32)

            y_pred = tensorrt_utils.do_inference(
                context,
                self.bindings,
                self.inputs,
                self.outputs,
                self.stream,
            )[0][0]

        return y_pred