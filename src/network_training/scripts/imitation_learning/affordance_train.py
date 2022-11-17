import torch
from torch import optim
from tqdm import tqdm

from imitation_learning import BaseTrain

class AffordanceTrain(BaseTrain):
    """
    Affordance Training Agent
    """
    def __init__(self,
                model,
                device,
                is_eval,
                train_params,
                log_params):
        
        super().__init__(model, device, is_eval, train_params, log_params)

    def configure(self, train_params, log_params):
        # filename
        self.checkpoint_filename = self.model_name + '_checkpoint.tar'
        self.model_filename      = self.model_name + '_model.pt'
        
        # optimizer      
        self.optimizer = optim.Adam(self.model.parameters(),
                                lr=train_params['optimizer']['learning_rate'],
                                betas=eval(train_params['optimizer']['betas']),
                                weight_decay=train_params['optimizer']['weight_decay'])

        # loss history
        self.loss_history = {
            'total_loss': [],
            'dist_loss': [],
            'angle_loss': [],
        }               

    def train(self):
        # Start training
        for epoch in tqdm(range(1, self.max_epochs + 1)):
            # Train
            self.model.train()
            train_total_loss = 0.0
            for _, batch_data in enumerate(self.train_dataloader):
                self.num_iter += 1
                batch_image = batch_data['image'].to(self.device)
                batch_y = batch_data['affordance'].to(self.device)
                batch_y_pred = self.model(batch_image)          
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
            
            # Test
            if self.test_dataloader is not None:
                test_total_loss = self.test()
            else:
                test_total_loss = 0

            # Update learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Logging
            self.epoch.append(epoch + self.last_epoch)
            tqdm.write('Epoch: {:d}, train loss = {:.3e} | test loss = {:.3e}'.
                        format(epoch + self.last_epoch,
                            train_total_loss, test_total_loss))
            
            if epoch % self.log_interval == 0:
                self.save_checkpoint(self.checkpoint_filename)
                self.save_model(self.model_filename)

        # End of training
        if self.use_tensorboard:
            self.writer.close()

    def test(self):
        self.model.eval()
        test_total_loss = 0.0
        for _, batch_data in enumerate(self.test_dataloader):
            batch_image = batch_data['image'].to(self.device)
            batch_y = batch_data['affordance'].to(self.device)
            batch_y_pred = self.model(batch_image)          
            test_loss = self.model.loss_function(batch_y_pred, batch_y)
            test_total_loss += test_loss['total_loss'].item()
        
        n_batch = len(self.test_dataloader)
        return test_total_loss / n_batch


class AffordanceCtrlTrain(BaseTrain):
    """
    Affordance Controller Training Agent
    """
    def __init__(self,
                model,
                afford_model,
                device,
                is_eval,
                train_params,
                log_params):
        
        # Load affordance model
        self.afford_model = afford_model.to(device)

        super().__init__(model, device, is_eval, train_params, log_params)

    def configure(self, train_params, log_params):
        # filename
        self.checkpoint_filename = self.model_name + '_checkpoint.tar'
        self.model_filename      = self.model_name + '_model.pt'
        
        # optimizer      
        self.optimizer = optim.Adam(self.model.parameters(),
                                lr=train_params['optimizer']['learning_rate'],
                                betas=eval(train_params['optimizer']['betas']),
                                weight_decay=train_params['optimizer']['weight_decay'])

        # loss history
        self.loss_history = {
            'total_loss': [],
        }               

    def train(self):
        # Start training
        for epoch in tqdm(range(1, self.max_epochs + 1)):
            # Train
            self.model.train()
            self.afford_model.eval()
            train_total_loss = 0
            for _, batch_data in enumerate(self.train_dataloader):
                self.num_iter += 1
                batch_image = batch_data['image'].to(self.device)
                batch_y = batch_data['action'].to(self.device)
                batch_affordance = batch_data['affordance'].to(self.device)
                # batch_affordance = self.afford_model(batch_image)
                batch_y_pred = self.model(batch_affordance).view(-1)       
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

            # Test
            if self.test_dataloader is not None:
                test_total_loss = self.test()
            else:
                test_total_loss = 0
    
            # Update learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step() 
                
            # Logging
            self.epoch.append(epoch + self.last_epoch)
            tqdm.write('Epoch: {:d}, train loss = {:.3e} | test loss = {:.3e}'.
                        format(epoch + self.last_epoch,
                            train_total_loss, test_total_loss))
            
            if epoch % self.log_interval == 0:
                self.save_checkpoint(self.checkpoint_filename)
                self.save_model(self.model_filename)

        # End of training
        if self.use_tensorboard:
            self.writer.close()

    def test(self):
        # Start testing
        self.model.eval()
        test_total_loss = 0.0
        with torch.no_grad():
            for _, batch_data in enumerate(self.test_dataloader):
                batch_image = batch_data['image'].to(self.device)
                batch_y = batch_data['affordance'].to(self.device)
                batch_y_pred = self.model(batch_image)
                test_loss = self.model.loss_function(batch_y_pred, batch_y)
                test_total_loss += test_loss['total_loss'].item()

            n_batch = len(self.test_dataloader)
            return test_total_loss / n_batch