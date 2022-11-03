import os, sys
import yaml
import numpy as np
import cv2
import torch
from torchvision import transforms

curr_dir        = os.path.dirname(os.path.abspath(__file__))
parent_dir      = os.path.dirname(curr_dir)
import_path     = os.path.join(parent_dir, '.')
sys.path.insert(0, import_path)

config_dir = os.path.join(parent_dir, 'configs')

from models import AffordanceNet

TRANSFORM_COMPOSED = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),
]) 

def read_yaml(file_path):
    '''
    Read yaml file
    '''
    file = open(file_path, 'r')
    config = yaml.safe_load(file)
    file.close()
    return config


class AffordanceCtrl():
    '''
    Affordance-Based Controller
    '''
    def __init__(self, **kwargs): 
        self.configure(**kwargs)
        self.load_model(**kwargs)

    def configure(self, **kwargs):
        '''
        Configure
        '''
        self.device             = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform_composed = TRANSFORM_COMPOSED
        
    def load_model(self, afford_model_path, ctrl_model_path=None, **kwargs):
        '''
        Load Model
        '''
        self.load_afford_model(afford_model_path)

    def load_afford_model(self, model_path, **kwargs):
        '''
        Load AffordanceNet Model
        '''
        afford_model_config = read_yaml(os.path.join(config_dir, 'affordance.yaml'))
        model = torch.load(model_path)
        self.afford_model = AffordanceNet(**afford_model_config['model_params']).to(self.device)
        self.afford_model.load_state_dict(model)
        self.afford_model.eval()
        self.image_resize = [self.afford_model.input_dim, self.afford_model.input_dim]
        
    def predict_affordance(self, image_color_list):
        '''
        Output predicted affordance
        '''
        
        if not isinstance(image_color_list, list):
            image_color_list = [image_color_list]
        
        image_tensor_list = []
        for image_color in image_color_list:
            image_np = image_color.copy()
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            image_np = cv2.resize(image_np, (self.image_resize[0], self.image_resize[1]))
            image_tensor = self.transform_composed(image_np)
            image_tensor_list.append(image_tensor)

        image_tensor_cat = torch.cat(tuple(image_tensor for image_tensor in image_tensor_list), dim=0)

        with torch.no_grad():
            out = self.afford_model(image_tensor_cat.unsqueeze(0).to(self.device))
            out = out.cpu().squeeze(0).numpy()
        
            affordance_pred = {
                'dist_center_width': out[0], # dist_center/width
                'rel_angle': out[1] * (np.pi/2), # rel_angle
            }

        return affordance_pred