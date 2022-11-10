import os, sys
import glob
import cv2
import pandas
import numpy as np
import random
import torch 
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from utils.train_utils import *
from models import AffordanceNet
from imitation_learning import AffordanceTrain

# Path settings
curr_dir    = os.path.dirname(os.path.abspath(__file__))
train_config_dir  = os.path.join(curr_dir, 'configs')
spline_data_dir  = os.path.join(curr_dir, 'ground_truth')

class SaltAndPepperNoise(object):
    def __init__(self,threshold = 0.05):
        self.threshold = threshold

    def __call__(self, img):
        img_transpose = torch.transpose(img,0,1)
        random_matrix = np.random.random(img_transpose.shape)
        img_transpose[random_matrix >= (1-self.threshold)] = 1
        img_transpose[random_matrix <= self.threshold] = 0
        return torch.transpose(img_transpose,1,0)

class MyTransform:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, sharpness=0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.salt_pepper_noise = SaltAndPepperNoise()
        self.sharpness = sharpness

    def __call__(self, img):
        img = transforms.functional.to_tensor(img)
        if random.random() > 0.5:
            img = transforms.functional.hflip(img)
            is_flip = True
        else:
            is_flip = False
        img = transforms.functional.rotate(img, np.random.uniform(-15,15))
        img = transforms.functional.gaussian_blur(img, random.choice([1,3,5,7]))
        img = self.salt_pepper_noise(img)
        img = transforms.functional.adjust_sharpness(img, np.random.uniform(max(0,1-self.sharpness),1+self.sharpness))
        img = transforms.functional.adjust_brightness(img, np.random.uniform(max(0,1-self.brightness),1+self.brightness))
        img = transforms.functional.adjust_contrast(img, np.random.uniform(max(0,1-self.contrast),1+self.contrast))
        img = transforms.functional.adjust_saturation(img, np.random.uniform(max(0,1-self.saturation),1+self.saturation))
        img = transforms.functional.normalize(img, (0.5), (0.5))
        return img, is_flip

class AffordanceDataset(Dataset):
    def __init__(self,
            dataset_dir,
            resize=None,
            affordance_dim=3,
            transform=None):

        self.rgb_file_list = []
        self.transform = transform
        self.resize = resize
        self.affordance_dim = affordance_dim
        self.affordance = np.empty((0, affordance_dim), dtype=np.float32)

        # Configure
        self.configure(dataset_dir)

    def configure(self, dataset_dir):
        for subfolder in os.listdir(dataset_dir):
            subfolder_path = os.path.join(dataset_dir, subfolder)
            print(subfolder_path)
            # RGB image
            rgb_file_list = glob.glob(os.path.join(subfolder_path, 'color', '*.png'))
            rgb_file_list.sort()
            affordance = self.get_affordance(subfolder_path)
            self.rgb_file_list.extend(rgb_file_list)
            self.affordance = np.concatenate((self.affordance, affordance), axis=0)
            
    def get_affordance(self, folder_path):
        data = pandas.read_csv(os.path.join(folder_path, 'pose.csv'))
        # Distance to centerline
        dist_center = data['dist_center'].to_numpy()
        # relative angle to centerline
        rel_angle = data['rel_angle'].to_numpy()
        # Output
        width = 6.0 # = 15 ft
        if self.affordance_dim == 3:
            result = np.column_stack([dist_center / width, # normalized
                                    rel_angle / (np.pi/2), 
                                    (dist_center + width / 2) / width - 0.5])
        else:
            result = np.column_stack([dist_center / width, # normalized
                                    rel_angle / (np.pi/2)])

        return result.astype(np.float32)

    def __len__(self):
        return len(self.rgb_file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Read RGB image
        bgr_img = cv2.imread(self.rgb_file_list[idx], cv2.IMREAD_UNCHANGED)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        if self.resize is not None:
            rgb_img = cv2.resize(rgb_img, (self.resize[0], self.resize[1]))
        if self.transform is not None:
            rgb_img, is_flip = self.transform(rgb_img)
        
        if is_flip:
            affordance = np.array(
                [-self.affordance[idx, j] for j in range(self.affordance_dim)],
                dtype=np.float32)
        else:
            affordance = self.affordance[idx,:]

        return {'image': rgb_img, 'affordance': affordance}

if __name__ == '__main__':
    # Read YAML configurations
    train_config = read_yaml(os.path.join(train_config_dir, 'train_config_field.yaml'))
    
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
    print('============== Affordance training ================')
    # Load parameters
    image_resize   = train_config['affordance_params']['image_resize']
    model_config   = read_yaml(os.path.join(train_config_dir, 'affordance' + '.yaml'))

    # Create the agent
    model = AffordanceNet(**model_config['model_params'])
    model_config['log_params']['output_dir'] = output_dir
    train_agent = AffordanceTrain(model=model,
                        device=device,
                        is_eval=False,
                        train_params=model_config['train_params'],
                        log_params=model_config['log_params'])

    # Warm start the model
    # train_agent.warm_start_model('/media/lab/Extreme SSD/my_outputs/warm_start/affordance_model.pt')

    # Random seed
    torch.manual_seed(model_config['train_params']['manual_seed'])
    torch.cuda.manual_seed(model_config['train_params']['manual_seed'])
    np.random.seed(model_config['train_params']['manual_seed'])

    # DataLoader
    all_data = AffordanceDataset(dataset_dir,
                resize=[image_resize[0],image_resize[1]],
                affordance_dim=model_config['model_params']['output_dim'],
                transform=MyTransform())

    print('Total length of data: ', str(len(all_data)))
    print(all_data.affordance[:,0].min(), all_data.affordance[:,0].max(), all_data.affordance[:,0].mean())
    print(all_data.affordance[:,1].min(), all_data.affordance[:,1].max(), all_data.affordance[:,1].mean())

    # Split the training and testing datasets
    if test_size == 0:
        # train_data = shuffle(all_data, random_state=random_state)
        train_data = all_data
        test_data = all_data
    else:
        train_data, test_data = train_test_split(all_data,
                                            test_size=test_size,
                                            random_state=random_state)       
    print('Loaded Affordance datasets successfully!')

    # Training loop
    print('\n*** Start training ***')
    train_agent.load_dataset(train_data, test_data)
    train_agent.train()
    print('Trained the model successfully.')