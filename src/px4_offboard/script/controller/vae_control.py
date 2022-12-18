import os
import cv2
import numpy as np
import yaml
import torch
from torchvision import transforms

from models import VanillaVAE, LatentCtrl

curr_dir        = os.path.dirname(os.path.abspath(__file__))
parent_dir      = os.path.dirname(curr_dir)
config_dir      = os.path.join(parent_dir, 'configs')

MAX_YAWRATE = 45.0 

def read_yaml(file_path):
    '''
    Read yaml file
    '''
    try:
        file = open(file_path, 'r')
        config = yaml.safe_load(file)
        file.close()
        return config

    except Exception as error:
        print(error)
        exit('Exit the program.')

class VAECtrl():
    '''
    VAE-Based Controller
    '''
    def __init__(self, **kwargs): 
        self.configure(**kwargs)
        self.load_model(**kwargs)
 
    def configure(self, **kwargs):
        '''
        Configure
        '''
        self.device             = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform_composed = transforms.Compose([ 
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5)),
                                ]) 
        
    def load_model(self, vae_model_path, latent_model_path, **kwargs):
        '''
        Load Model
        '''
        self.load_VAE_model(vae_model_path)
        self.load_latent_model(latent_model_path)

    def load_VAE_model(self, model_path):
        '''
        Load VAE model
        '''
        model_config = {
            
        }
        model = torch.load(model_path)
        self.VAE_model = VanillaVAE(**model_config['model_params']).to(self.device)
        self.VAE_model.load_state_dict(model)
        self.z_dim = self.VAE_model.z_dim
        self.image_resize = [self.VAE_model.input_dim, self.VAE_model.input_dim]

    def load_latent_model(self, model_path):
        '''
        Load Latent FC model
        '''
        model_config = read_yaml(os.path.join(config_dir, 'latent_ctrl.yaml'))
        model = torch.load(model_path)
        model_config['model_params']['z_dim'] = self.z_dim
        self.Latent_model = LatentCtrl(**model_config['model_params']).to(self.device)
        self.Latent_model.load_state_dict(model)

    def predict(self, image_color, is_bgr=True, state_extra=None):
        '''
        Predict the action
        '''
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
        
        # # Apply a filter
        # alpha = 0.5
        # y_pred = alpha * y_pred + (1 - alpha) * cmd_history[-1]

        return y_pred

    def reconstruct_image(self, image_color, is_bgr=True):
        '''
        Reconstruct the images from VAE for visualization
        '''
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