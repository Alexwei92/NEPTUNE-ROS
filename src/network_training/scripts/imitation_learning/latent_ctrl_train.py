import torch
from torch import optim
from tqdm import tqdm

from imitation_learning import BaseTrain

class LatentCtrlTrain(BaseTrain):
    '''
    Latent Controller Training Agent
    '''
    def __init__(self,
                model,
                VAE_model,
                device,
                is_eval,
                train_params,
                log_params):
        
        # Load VAE model
        self.VAE_model = VAE_model.to(device)
        
        super().__init__(model, device, is_eval, train_params, log_params)

    def configure(self, train_params, log_params):
        # z dimension check
        self.z_dim = self.model.get_latent_dim()
        if self.VAE_model.get_latent_dim() != self.z_dim:
            raise Exception('z_dim does not match!')

        # filename
        self.model_name          = self.model_name + '_' + self.VAE_model.name
        self.checkpoint_filename = self.model_name + '_checkpoint_z_' + str(self.z_dim) + '.tar'
        self.model_filename      = self.model_name + '_model_z_' + str(self.z_dim) + '.pt'

        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(),
                                lr=train_params['optimizer']['learning_rate'],
                                betas=eval(train_params['optimizer']['betas']),
                                weight_decay=train_params['optimizer']['weight_decay'])

        # loss history
        self.loss_history = {
            'total_loss': []
        }

    def train(self):
        # Start training
        for epoch in tqdm(range(1, self.max_epochs+1)):
            # Train
            self.model.train()
            self.VAE_model.eval() # freeze the VAE model
            train_total_loss = 0
            for _, batch_data in enumerate(self.train_dataloader):
                self.num_iter += 1
                batch_image = batch_data['image'].to(self.device)
                batch_y = batch_data['action'].to(self.device)
                if 'state_extra' in batch_data:
                    batch_extra = batch_data['state_extra'].to(self.device)
                else:
                    batch_extra = None
                
                batch_z = self.VAE_model.get_latent(batch_image, with_logvar=True)
                batch_y_pred = self.model(batch_z, batch_extra).view(-1)
                train_loss = self.model.loss_function(batch_y_pred, batch_y)
                self.optimizer.zero_grad()
                train_loss['total_loss'].backward()
                self.optimizer.step()

                train_total_loss += train_loss['total_loss'].item()
                self.iteration.append(self.num_iter)
                for name in train_loss.keys():
                    self.loss_history[name].append(train_loss[name].item())               

                if self.use_tensorboard:
                    for name in train_loss.keys():
                        self.writer.add_scalar('Train/' + name, train_loss[name].item(), self.num_iter)
            
            n_batch = len(self.train_dataloader)
            train_total_loss /= n_batch

            # # Test
            # if self.test_dataloader is not None:
            #     test_total_loss = self.test()

            # Update learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # logging
            self.epoch.append(epoch + self.last_epoch)
            tqdm.write('Epoch: {:d}, train loss = {:.3e}'.format(epoch + self.last_epoch, train_total_loss))

            if epoch % self.log_interval == 0:
                self.save_checkpoint(self.checkpoint_filename)
                self.save_model(self.model_filename)

        # End of training
        if self.use_tensorboard:
            self.writer.close()
            
    def test(self):
        self.model.eval()
        self.VAE_model.eval()
        test_total_loss = 0
        with torch.no_grad():
            for _, batch_data in enumerate(self.test_dataloader):
                batch_image = batch_data['image'].to(self.device)
                batch_y = batch_data['action'].to(self.device)
                if 'extra' in batch_data:
                    batch_extra = batch_data['extra'].to(self.device)
                else:
                    batch_extra = None

                batch_z = self.VAE_model.get_latent(batch_image, with_logvar=False)
                batch_y_pred = self.model(batch_z, batch_extra).view(-1)
                test_loss = self.model.loss_function(batch_y_pred, batch_y)
                test_total_loss += test_loss['total_loss'].item()
        
        n_batch = len(self.test_dataloader)
        test_total_loss /= n_batch
        return test_total_loss