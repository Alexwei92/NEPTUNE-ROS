import os
import sys
import cv2
import numpy as np
import torch
from torchvision import transforms

curr_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.abspath(os.path.join(curr_dir, "../../../network_training/scripts"))
sys.path.insert(0, model_dir)

from models import VAELatentCtrl

vae_latent_ctrl_model_config = {
    'name': 'vae_latent_ctrl',
    'input_dim': 128,
    'in_channels': 3,
    'z_dim': 1000,
    'extra_dim': 5,
}

class VAELatentController():
    '''
    VAE-Based Controller with reduced size VAE and LatentCtrl
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
        
    def load_model(self, model_weight_path, **kwargs):
        '''
        Load Model
        '''
        self.model = VAELatentCtrl(**vae_latent_ctrl_model_config).to(self.device)
        model_weight = torch.load(model_weight_path)
        self.model.load_state_dict(model_weight)

    def predict(self, image_color, is_bgr=True, state_extra=None):
        '''
        Predict the action
        '''
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