import os
import numpy as np
import cv2
import glob
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torchvision import transforms
from torch.utils.data import Dataset

from utils.train_utils import read_yaml
from models import VAEGAN
from imitation_learning import VAEGANTrain

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
# torch.autograd.set_detect_anomaly(True)

####################################################
transform_composed = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAdjustSharpness(np.random.uniform(1-0.3, 1+0.3)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.0),
    transforms.Normalize((0.5), (0.5))
])
###################################################

# Path settings
curr_dir    = os.path.dirname(os.path.abspath(__file__))
train_config_dir  = os.path.join(curr_dir, 'configs')

class VAEDataset(Dataset):
    def __init__(self,
            dataset_dir,
            resize=None,
            transform=None):

        self.rgb_file_list = []
        self.transform = transform
        self.resize = resize

        # Configure
        self.configure(dataset_dir)

    def configure(self, dataset_dir):
        for folder in dataset_dir:
            for subfolder in os.listdir(folder):
                subfolder_path = os.path.join(folder, subfolder)
                print(subfolder_path)
                # RGB image
                rgb_file_list = glob.glob(os.path.join(subfolder_path, 'color', '*.png'))
                rgb_file_list.sort()
                self.rgb_file_list.extend(rgb_file_list)

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
            rgb_img = self.transform(rgb_img)

        return {'image': rgb_img}

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

    # Extra data folder
    extra_dataset_dir   = train_config['path_params']['extra_dataset_dir']
    if extra_dataset_dir != 'None':
        dataset_dir = [dataset_dir, extra_dataset_dir]
    else:
        dataset_dir = [dataset_dir]

    # Load training settings
    device              = torch.device(train_config['train_params']['device'])

    # Load Dataloader settings
    test_size           = train_config['dataset_params']['test_size']
    random_state        = train_config['dataset_params']['random_state']

    ##########  Training   ###########
    print('============== VAE GAN training ================')
    # Load parameters
    model_config   = read_yaml(os.path.join(train_config_dir, 'vae_gan.yaml'))

    # Create the agent
    model = VAEGAN(**model_config['model_params'])
    model_config['log_params']['output_dir'] = output_dir
    train_agent = VAEGANTrain(model=model,
                        device=device,
                        is_eval=False,
                        train_params=model_config['train_params'],
                        log_params=model_config['log_params'])

    # Random seed
    torch.manual_seed(model_config['train_params']['manual_seed'])
    torch.cuda.manual_seed(model_config['train_params']['manual_seed'])
    np.random.seed(model_config['train_params']['manual_seed'])

    # Load data   
    image_resize = [model_config['model_params']['input_dim'], model_config['model_params']['input_dim']]
    all_data = VAEDataset(dataset_dir,
                        resize=image_resize,
                        transform=transform_composed)

    # Split the training and testing datasets
    if test_size == 0:
        # train_data = shuffle(all_data, random_state=random_state)
        train_data = all_data
        test_data = all_data
    else:
        train_data, test_data = train_test_split(all_data,
                                            test_size=test_size,
                                            random_state=random_state)       
    print('Loaded VAE GAN datasets successfully!')
    print('Total number of images = %d' % len(train_data))

    # Training loop
    print('\n*** Start training ***')
    train_agent.load_dataset(train_data, test_data)
    train_agent.train()
    print('Trained the model successfully.')