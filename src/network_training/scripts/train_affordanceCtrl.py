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

from models import AffordanceFC, AffordanceNet
from imitation_learning import AffordanceCtrlTrain
from utils.train_utils import *

# Path settings
curr_dir    = os.path.dirname(os.path.abspath(__file__))
train_config_dir  = os.path.join(curr_dir, 'configs')
spline_data_dir  = os.path.join(curr_dir, 'ground_truth')

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
# torch.autograd.set_detect_anomaly(True)

####################################################
MAX_YAWRATE = 45.0
transform_composed = transforms.Compose([
    transforms.ToTensor(),
    # transforms.RandomAdjustSharpness(np.random.uniform(0.8,1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.Normalize((0.5), (0.5)),
])
###################################################
class MyTransform:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, sharpness=0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        # self.sharpness = sharpness

    def __call__(self, img):
        img = transforms.functional.to_tensor(img)
        if random.random() > 0.5:
            img = transforms.functional.hflip(img)
            is_flip = True
        else:
            is_flip = False
        # img = transforms.functional.adjust_sharpness(img, np.random.uniform(max(0,1-self.sharpness),1+self.sharpness))
        img = transforms.functional.adjust_brightness(img, np.random.uniform(max(0,1-self.brightness),1+self.brightness))
        img = transforms.functional.adjust_contrast(img, np.random.uniform(max(0,1-self.contrast),1+self.contrast))
        img = transforms.functional.adjust_saturation(img, np.random.uniform(max(0,1-self.saturation),1+self.saturation))
        img = transforms.functional.normalize(img, (0.5), (0.5))
        return img, is_flip

class AffordanceCrtlDataset(Dataset):
    def __init__(self,
            dataset_dir,
            with_yawRate=False,
            resize=None,
            map_data=None,
            transform=None):

        self.rgb_file_list = []
        self.transform = transform
        self.resize = resize
        self.map_data = map_data
        self.affordance = np.empty((0, 3), dtype=np.float32)
        self.output = np.empty((0,), dtype=np.float32)
        if with_yawRate:
            self.state_extra = np.empty((0, 1), dtype=np.float32)
        else:
            self.state_extra = None

        # Configure
        self.configure(dataset_dir, with_yawRate)

    def configure(self, dataset_dir, with_yawRate):
        for subfolder in os.listdir(dataset_dir):
            subfolder_path = os.path.join(dataset_dir, subfolder)
            print(subfolder_path)
            # Telemetry
            state_extra, output, is_pilot, affordance = self.read_telemetry(subfolder_path, with_yawRate)
            if output is not None:
                # RGB image
                rgb_file_list = glob.glob(os.path.join(subfolder_path, 'color', '*.png'))
                rgb_file_list.sort()
                max_index = min(len(rgb_file_list),len(output))
                is_pilot = is_pilot[:max_index]
                rgb_file_list = np.array(rgb_file_list[:max_index])[is_pilot]
                self.rgb_file_list.extend(rgb_file_list.tolist())
                self.output = np.concatenate((self.output, output[:max_index][is_pilot]), axis=0)
                if state_extra is not None:
                    self.state_extra = np.concatenate((self.state_extra, state_extra[:max_index,:][is_pilot,:]), axis=0)                    
                self.affordance = np.concatenate((self.affordance, affordance[:max_index,:][is_pilot]), axis=0)

    def read_telemetry(self, folder_dir, with_yawRate):
        telemetry_data = pandas.read_csv(os.path.join(folder_dir, 'airsim.csv'))
        N = len(telemetry_data) - 1 # length of data 

        if telemetry_data.iloc[-1,0] == 'crashed':
            print('Find crashed dataset in {:s}'.format(folder_dir))
            N -= 20 # remove the last # of data
            if N < 0:
                return None, None, None, None

        # Yaw cmd
        y = telemetry_data['yaw_cmd'][:N].to_numpy(dtype=np.float32)

        # Yaw Rate
        X_extra = None
        if with_yawRate:
            yawRate = np.reshape(telemetry_data['yaw_rate'][:N].to_numpy(dtype=np.float32), (-1,1))
            yawRate_norm = yawRate * (180.0 / math.pi) / MAX_YAWRATE
            X_extra = yawRate_norm

        # flag
        flag = telemetry_data['flag'][:N].to_numpy()
        is_pilot = (flag == 0)
        
        # ground truth affordance 
        pilot_data = read_pilot_data(os.path.join(folder_dir, 'airsim.csv'))
        affordance = calculate_affordance(self.map_data, pilot_data)
        # Distance to the sides
        dist_left, dist_right  = affordance['dist_left'], affordance['dist_right']
        # Distance to centerline
        dist_center = affordance['dist_center']
        # relative angle to centerline
        rel_angle = affordance['rel_angle']
        # Output
        width = dist_left + dist_right
        affordance = np.column_stack([dist_center / width, # normalized
                                rel_angle / (np.pi/2), 
                                dist_left / width - 0.5]).astype(np.float32)

        return X_extra, y, is_pilot, affordance


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
        
        if is_flip:
            a = np.float32(-1) 
        else:
            a = np.float32(1)
        
        if self.state_extra is None:
            return {'image': rgb_img, 'action': a*self.output[idx], 'affordance': a*self.affordance[idx,:]}
        else:
            return {'image': rgb_img, 'extra': a*self.state_extra[idx, :], 'action': a*self.output[idx], 'affordance': a*self.affordance[idx,:]}

if __name__ == '__main__':

    ###########  Load parameters   ###########
    # Read YAML configurations
    train_config = read_yaml(os.path.join(train_config_dir, 'train_config.yaml'))

    # Load path settings
    dataset_dir = train_config['path_params']['dataset_dir']
    output_dir  = train_config['path_params']['output_dir']

    if not os.path.isdir(dataset_dir):
        raise IOError("No such folder {:s}".format(dataset_dir))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Load training settings
    device              = torch.device(train_config['train_params']['device'])

    # Load Dataloader settings
    test_size           = train_config['dataset_params']['test_size']
    random_state        = train_config['dataset_params']['random_state']

    ##########  Training   ###########
    print('============== Affordance NN Controller ================')
    # Load parameters
    image_resize        = train_config['affordanceCtrl_params']['image_resize']
    model_path          = train_config['affordanceCtrl_params']['model_path']
    afford_model_config = read_yaml(os.path.join(train_config_dir, 'affordance.yaml'))
    ctrl_model_config   = read_yaml(os.path.join(train_config_dir, 'affordance_ctrl.yaml'))
    with_yawRate        = ctrl_model_config['model_params']['with_yawRate']
    map_name            = train_config['affordanceCtrl_params']['map_name']
    map_data            = read_map_data(os.path.join(spline_data_dir, map_name+'_spline_result.csv'))

    # Load affordance model
    afford_model = AffordanceNet(**afford_model_config['model_params'])

    if not os.path.isfile(model_path):
        raise IOError("***No such file!", model_path)
    else:
        model = torch.load(model_path)
        afford_model.load_state_dict(model)
                
    # Create the agent
    ctrl_model_config['log_params']['output_dir'] = output_dir
    ctrl_model = AffordanceFC(**ctrl_model_config['model_params'])

    train_agent = AffordanceCtrlTrain(model=ctrl_model,
                        afford_model=afford_model,
                        device=device,
                        is_eval=False,
                        train_params=ctrl_model_config['train_params'],
                        log_params=ctrl_model_config['log_params'])

    # Warm start the model
    # train_agent.warm_start_model('/media/lab/Extreme SSD/my_outputs/longmap1/affordance_ctrl/iter0/affordance_ctrl/affordance_ctrl_model.pt')

    # Random seed
    torch.manual_seed(ctrl_model_config['train_params']['manual_seed'])
    torch.cuda.manual_seed(ctrl_model_config['train_params']['manual_seed'])
    np.random.seed(ctrl_model_config['train_params']['manual_seed'])

    # Load data
    print('Loading datasets from {:s}'.format(dataset_dir))
    all_data = AffordanceCrtlDataset(dataset_dir,
                    with_yawRate=with_yawRate,
                    resize=image_resize,
                    map_data=map_data,
                    transform=MyTransform())

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

    # Training loop
    print('\n*** Start training ***')
    train_agent.load_dataset(train_data, test_data)
    train_agent.train()
    print('Trained AffordanceCtrl model successfully.')

###################





