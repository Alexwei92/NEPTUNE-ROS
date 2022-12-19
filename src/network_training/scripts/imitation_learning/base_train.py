import os
import datetime
import torch
import shutil
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

SAMPLE_SIZE = 64 # number of images in the sample
NUM_WORKER  = 4 # number of workers in parallel


def genereate_sample_folder(folder_path, dataloader, checkpoint_preload):
    '''
    Generate a sample folder
    '''
    if not checkpoint_preload and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
        _, example_data = next(enumerate(dataloader))
        example_data = example_data['image']
        torch.save(example_data, os.path.join(folder_path, 'sample_image_data.pt'))
        vutils.save_image(example_data,
                        os.path.join(folder_path, 'sample_image.png'),
                        normalize=True,
                        range=(-1,1))
    else:
        example_data = torch.load(os.path.join(folder_path, 'sample_image_data.pt'))
    
    return example_data


class BaseTrain():
    """
    Base Training Agent
    """
    def __init__(self,
                model,
                device,
                is_eval,
                train_params,
                log_params):
        
        # Model
        self.device = device
        self.model  = model.to(device)

        # Dataloader
        self.train_dataloader = None
        self.test_dataloader  = None
        self.validation_data  = None

        # Training parameters
        self.max_epochs = train_params['n_epochs']
        self.batch_size = train_params['batch_size']
       
        # Logging parameters
        self.last_epoch     = 0
        self.num_iter       = 0
        self.epoch          = []
        self.iteration      = []
        self.tb_folder_name = datetime.datetime.now().strftime("%Y_%h_%d_%H_%M_%S")

        self.checkpoint_preload = log_params['checkpoint_preload']
        self.log_interval       = log_params['log_interval']
        self.use_tensorboard    = log_params['use_tensorboard']
        self.model_name         = log_params['name']
        self.log_folder         = os.path.join(log_params['output_dir'], self.model_name)
        if not os.path.isdir(self.log_folder):
            os.makedirs(self.log_folder)

        if 'generate_samples' in log_params:
            self.generate_samples   = log_params['generate_samples']
        else:
            self.generate_samples   = False

        # Filename, optimizer and loss history configure
        self.configure(train_params, log_params)

        # Load a checkpoint
        self.checkpoint_filename    = os.path.join(self.log_folder, self.checkpoint_filename)
        self.model_filename         = os.path.join(self.log_folder, self.model_filename)
        
        if self.generate_samples:
            self.sample_folder_path = os.path.join(self.log_folder, self.sample_folder_path)
        
        if is_eval or self.checkpoint_preload:
            self.load_checkpoint(self.checkpoint_filename)

        if train_params['lr_scheduler']['enable']:
            if self.checkpoint_preload:
                self.lr_scheduler = StepLR(self.optimizer, step_size=train_params['lr_scheduler']['step_size'],
                                    gamma=train_params['lr_scheduler']['gamma'], last_epoch=self.last_lr_scheduler.last_epoch)
            else:
                self.lr_scheduler = StepLR(self.optimizer, step_size=train_params['lr_scheduler']['step_size'],
                                    gamma=train_params['lr_scheduler']['gamma'])
        else:
            self.lr_scheduler = None

        # Use tensorboard
        if self.use_tensorboard:
            self.writer = SummaryWriter(os.path.join(self.log_folder, 'tb/' + self.tb_folder_name))

    def load_checkpoint(self, file_path):
        if not os.path.isfile(file_path):
            raise IOError("***No such file!", file_path)

        checkpoint = torch.load(file_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch          = checkpoint['epoch']
        self.iteration      = checkpoint['iteration']
        self.loss_history   = checkpoint['loss_history']
        self.tb_folder_name = checkpoint['tb_folder_name']
        self.last_epoch     = self.epoch[-1]
        self.num_iter       = self.iteration[-1]
        if 'lr_scheduler' in checkpoint:
            self.last_lr_scheduler = checkpoint['lr_scheduler']
  
    def load_train_dataset(self, train_data, is_shuffle=True):
        if train_data is not None:
            self.train_dataloader = DataLoader(train_data,
                                    batch_size=self.batch_size,
                                    shuffle=is_shuffle,
                                    num_workers=NUM_WORKER,
                                    drop_last=False,
                                    pin_memory=True)
        else:
            raise Exception('No training data found!')

    def load_test_dataset(self, test_data, is_shuffle=True):
        if test_data is not None:
            self.test_dataloader = DataLoader(test_data,
                                    batch_size=self.batch_size,
                                    shuffle=is_shuffle,
                                    num_workers=NUM_WORKER,
                                    drop_last=False)
            # Generate sample folder
            if self.generate_samples:
                sample_dataloader = DataLoader(test_data,
                                    batch_size=SAMPLE_SIZE,
                                    shuffle=is_shuffle,
                                    num_workers=NUM_WORKER,
                                    drop_last=True)
                self.validation_data = genereate_sample_folder(self.sample_folder_path, sample_dataloader, self.checkpoint_preload)
        else:
            self.generate_samples = False

    def load_dataset(self, train_data, test_data):
        self.load_train_dataset(train_data)
        self.load_test_dataset(test_data) 

    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)

    def save_sample(self, epoch, validation_data, sample_folder_path):
        self.model.eval()
        with torch.no_grad():
            results = self.model(validation_data.to(self.device))
            vutils.save_image(results[0],
                              os.path.join(sample_folder_path, 'reconstructed_image_epoch_{:d}.png'.format(epoch)),
                              normalize=True,
                              range=(-1, 1))
            if self.use_tensorboard:
                img_grid = vutils.make_grid(results[0], normalize=True, range=(-1, 1))
                self.writer.add_image('reconstructed_image', img_grid, epoch)

    def get_current_epoch(self):
        return self.epoch[-1]

    def get_current_iteration(self):
        return self.iteration[-1]

    def get_train_history(self):
        return self.iteration, self.loss_history

    def get_log_folder(self):
        return self.log_folder
        
    def configure(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
 
    def test(self):
        raise NotImplementedError

    def save_checkpoint(self, file_path):
        checkpoint_dict = {
            'epoch': self.epoch,
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'tb_folder_name': self.tb_folder_name,
        }
        torch.save(checkpoint_dict, file_path)

    def warm_start_model(self, model_path):
        model_weight = torch.load(model_path)
        self.model.load_state_dict(model_weight)
        print('Warm start the model from {:s} successfully.'.format(model_path))