import os
import numpy as np
import glob
import cv2
import torch
import random
import pandas
import math
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torchvision import transforms
from torch.utils.data import Dataset

from models import EndToEnd
from imitation_learning import EndToEndTrain
from utils.train_utils import *

# Path settings
curr_dir    = os.path.dirname(os.path.abspath(__file__))
train_config_dir  = os.path.join(curr_dir, 'configs')

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
# torch.autograd.set_detect_anomaly(True)

####################################################

class SaltAndPepperNoise(object):
    def __init__(self,threshold = 0.01):
        self.threshold = threshold

    def __call__(self, img):
        img_transpose = torch.transpose(img,0,1)
        random_matrix = np.random.random(img_transpose.shape)
        img_transpose[random_matrix >= (1-self.threshold)] = 1
        img_transpose[random_matrix <= self.threshold] = 0
        return torch.transpose(img_transpose,1,0)

class GaussianNoise(object):
    def __init__(self, mean=0, std = 0.05):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        noise = np.random.normal(self.mean, self.std, img.shape)
        img = img + noise.astype(np.float32)
        return torch.clamp(img, 0, 1)

class MyTransform:
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, sharpness=0.3):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.sharpness = sharpness
        self.salt_pepper_noise = SaltAndPepperNoise()
        self.gaussian_noise = GaussianNoise(0, 0.05)

    def __call__(self, img):
        img = transforms.functional.to_tensor(img)
        if random.random() > 0.5:
            img = transforms.functional.hflip(img)
            is_flip = True
        else:
            is_flip = False

        img = transforms.functional.rotate(img, np.random.uniform(-10,10))
        img = transforms.functional.gaussian_blur(img, random.choice([1,3,5]))
        # img = self.salt_pepper_noise(img)
        img = self.gaussian_noise(img)
        img = transforms.functional.adjust_sharpness(img, np.random.uniform(max(0,1-self.sharpness),1+self.sharpness))
        img = transforms.functional.adjust_brightness(img, np.random.uniform(max(0,1-self.brightness),1+self.brightness))
        img = transforms.functional.adjust_contrast(img, np.random.uniform(max(0,1-self.contrast),1+self.contrast))
        img = transforms.functional.adjust_saturation(img, np.random.uniform(max(0,1-self.saturation),1+self.saturation))
        img = transforms.functional.normalize(img, (0.5), (0.5))
        return img, is_flip

class EndToEndDataset(Dataset):
    def __init__(self,
            dataset_dir,
            iteration=0,
            resize=None,
            transform=None,
            enable_extra=True):

        self.rgb_file_list = []
        self.transform = transform
        self.resize = resize
        self.enable_extra = enable_extra
        self.action = np.empty((0,), dtype=np.float32)
        self.state_extra = np.empty((0, 6), dtype=np.float32)

        # Configure
        self.configure(dataset_dir, iteration)

    def configure(self, dataset_dir, iteration):
        for iter in range(iteration+1):
            folder_path = os.path.join(dataset_dir, 'iter' + str(iter))
            for subfolder in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder)
                print(subfolder_path)
                # Mavros
                state_extra, action, is_pilot = self.read_mavros_data(subfolder_path)
                # RGB image
                rgb_file_list = glob.glob(os.path.join(subfolder_path, 'color', '*.png'))
                rgb_file_list.sort()
                rgb_file_list = np.array(rgb_file_list)[is_pilot]
                self.rgb_file_list.extend(rgb_file_list.tolist())
                self.action = np.concatenate((self.action, action[is_pilot]), axis=0)
                if state_extra is not None:
                    self.state_extra = np.concatenate((self.state_extra, state_extra[is_pilot,:]), axis=0)

    def read_mavros_data(self, folder_dir):
        mavros_data = pandas.read_csv(os.path.join(folder_dir, 'states.csv'))
        N = len(mavros_data) # length of data 

        # angles
        roll = mavros_data['roll_rad'].to_numpy(dtype=np.float32)
        pitch = mavros_data['pitch_rad'].to_numpy(dtype=np.float32)

        # velocity body
        linear_x = mavros_data['body_linear_x'].to_numpy(dtype=np.float32)
        linear_y = mavros_data['body_linear_y'].to_numpy(dtype=np.float32)
        linear_z = mavros_data['body_linear_z'].to_numpy(dtype=np.float32)
        angular_x = mavros_data['body_angular_x'].to_numpy(dtype=np.float32)
        angular_y = mavros_data['body_angular_y'].to_numpy(dtype=np.float32)
        angular_z = mavros_data['body_angular_z'].to_numpy(dtype=np.float32)

        # relative height
        relative_height = mavros_data['odom_rel_height'].to_numpy(dtype=np.float32)

        # control cmd
        action = mavros_data['control_cmd'].to_numpy(dtype=np.float32)

        # ai mode
        ai_mode = mavros_data['ai_mode'].to_numpy()

        # extra states array
        state_extra = np.array((
            roll,
            pitch,
            linear_x,
            linear_y,
            # linear_z,
            # angular_x,
            # angular_y,
            angular_z,
            relative_height,
        )).T

        # flag
        is_pilot = (~ai_mode)

        return state_extra, action, is_pilot


    def __len__(self):
        return len(self.rgb_file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Read RGB image
        rgb_img = cv2.imread(self.rgb_file_list[idx], cv2.IMREAD_UNCHANGED)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        if self.resize is not None:
            rgb_img = cv2.resize(rgb_img, (self.resize[0], self.resize[1]))
        if self.transform is not None:
            rgb_img, is_flip = self.transform(rgb_img)
               
        # if enable state extra
        if self.enable_extra:
            state_extra = self.state_extra[idx, :]
        else:
            state_extra = np.zeros(6, dtype=np.float32)

        # if flip
        if is_flip:
            a = np.float32(-1)
            state_extra_output = np.array([-1, 1, 1, -1, -1, 1], dtype=np.float32) * state_extra
        else:
            a = np.float32(1)
            state_extra_output = state_extra
            
        if self.state_extra is None:
            return {'image': rgb_img, 'action': a*self.action[idx]}
        else:
            return {'image': rgb_img, 'state_extra': state_extra_output, 'action': a*self.action[idx]}

if __name__ == '__main__':

    ###########  Load parameters   ###########
    # Read YAML configurations
    train_config = read_yaml(os.path.join(train_config_dir, 'train_config_field.yaml'))

    # Load path settings
    dataset_dir = '/media/lab/NEPTUNE2/field_datasets/human_data'
    output_dir  = '/media/lab/NEPTUNE2/field_outputs/imitation_learning'

    if not os.path.isdir(dataset_dir):
        raise IOError("No such folder {:s}".format(dataset_dir))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Load training settings
    device              = torch.device(train_config['train_params']['device'])

    # Load Dataloader settings
    test_size           = train_config['dataset_params']['test_size']
    random_state        = train_config['dataset_params']['random_state']
    iteration           = train_config['dataset_params']['iteration']

    ##########  Training   ###########
    print('============== End to End Controller ================')
    # Load parameters
    model_config        = read_yaml(os.path.join(train_config_dir, 'end_to_end.yaml'))

              
    # Create the agent
    model_config['log_params']['output_dir'] = os.path.join(output_dir, 'iter'+str(iteration))
    model = EndToEnd(**model_config['model_params'])

    train_agent = EndToEndTrain(model=model,
                        device=device,
                        is_eval=False,
                        train_params=model_config['train_params'],
                        log_params=model_config['log_params'])

    # Random seed
    torch.manual_seed(model_config['train_params']['manual_seed'])
    torch.cuda.manual_seed(model_config['train_params']['manual_seed'])
    np.random.seed(model_config['train_params']['manual_seed'])

    # Load data
    print('Loading datasets from {:s}'.format(dataset_dir))
    image_resize = [model_config['model_params']['input_dim'], model_config['model_params']['input_dim']]
    enable_extra = model_config['model_params']['enable_extra']
    all_data = EndToEndDataset(dataset_dir,
                    iteration=iteration,
                    resize=image_resize,
                    transform=MyTransform(),
                    enable_extra=enable_extra)

    # Split the training and testing datasets
    if test_size == 0:
        # train_data = shuffle(all_data, random_state=random_state)
        train_data = all_data
        test_data = all_data
    else:
        train_data, test_data = train_test_split(all_data,
                                            test_size=test_size,
                                            random_state=random_state)       
    print('Loaded datasets successfully!')
    print('Total number of data = %d' % len(train_data))

    # Training loop
    print('\n*** Start training ***')
    train_agent.load_dataset(train_data, test_data)
    train_agent.train()
    print('Trained EndToEnd model successfully.')

###################