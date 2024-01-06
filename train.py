import torch
import copy
import random
from dataclasses import dataclass
from typing import Any, List, Tuple

# training config
@dataclass
class TrainConfig:
    model: torch.nn
    train_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
    epochs: int
    device: torch.device
    criterion: torch.nn
    optimizer: torch.optim
    patience: int


class Trainer:
    def __init__(self, config: TrainConfig):   

        # unpack config
        self.model = config.model
        self.train_loader = config.train_loader
        self.val_loader = config.test_loader
        self.epochs = config.epochs
        self.device = config.device
        self.criterion = config.criterion
        self.optimizer = config.optimizer
        self.patience = config.patience


    def train(self):
    
        #send model to device
        self.model.to(self.device)
        print(f"Model sent to {self.device}")


        for epoch in range(self.epochs):
            
            # set model to train mode
            self.model.train() 

     
            train_iterator = iter(self.train_loader)                    # reset train iterator
            
            train_iterator_list = list(train_iterator)                  # convert iterator to list to apply random.shuffle()
            random.shuffle(train_iterator_list)                         # shuffle the list of batches
        
            train_iterator = iter(train_iterator_list)                  # convert shuffled iterator-list back to iterator
            
            half_of_num_batches = int(len(train_iterator_list) / 2)     # compute half of the num batches for the training loop
            
            for i in range(half_of_num_batches): #  <-------------------- Iterate through each 1/2 the batches of the 
                                                 #   <------------------- trainloader since 2 batches pulled per inference
                
                # handle dataset containing non-even batch ----> Stop-Iteration Error
                try:
                    imgs_1, targets_1 = next(train_iterator)  # get next two batchs of the iterator
                    imgs_2, targets_2 = next(train_iterator)
                except:                                       # if there are is only one batch left, that means we are at 
                    break                                     # the end of the iterator and can break out to the next epoch

                
                imgs_1, targets_1 = imgs_1.to(self.device), targets_1.to(self.device)   # Load images to GPU
                imgs_2, targets_2 = imgs_2.to(self.device), targets_2.to(self.device)
                
                self.optimizer.zero_grad()                                              #  Clear gradients

                image_1_embedding, image_2_embedding = self.model(imgs_1, imgs_2)       # Forward pass  ---> 2 vector embeddings
                
                try:
                    targets = (targets_1 == targets_2).float()                          # create binary target vector by comparing the labels of the two images
                except:
                    break                                                               # if there are is only one batch left, that means we are at 
                                                                                        # the end of the iterator and can break out to the next epoch

                targets = targets.to(self.device)                                       # send target to device

                loss = self.criterion(image_1_embedding, image_2_embedding, targets)    # compute loss 
                train_loss += loss.item()                                               # add loss to running total 

                loss.backward()                                                         # back prop

                self.optimizer.step()                                                   # update weights
                
            train_loss /= half_of_num_batches                                           # Average loss sum across the batch
            self.historical_train_loss.append(train_loss)                               # save loss for plotting

            val_loss = self.validate()                                                  # compute validation loss
           
            print(f"Epoch {epoch}: Train Loss: {train_loss:.5f} -- Val Loss: {val_loss:.5f}")
  
            self.early_stopping(epoch, val_loss)                                        # check if early stopping criteria has been met

            if self.es_counter >= self.patience:                        # break train loop if early stopping criteria has been met
                break
          
        return self.best_model, self.historical_loss, self.historical_train_loss

            
    def validate(self):
            
        self.model.eval()
        
        with torch.no_grad():
             
            val_iterator = iter(self.val_loader)      # reset val iterator
            val_loss = 0                              # initialize val loss sum 

            for i in range(len(val_iterator)):

                # get the next batch of images and targets from the validation set
                imgs_1, targets_1 = next(val_iterator)
                imgs_2, targets_2 = next(val_iterator)

                # Load images to GPU
                imgs_1, targets_1 = imgs_1.to(self.device), targets_1.to(self.device)
                imgs_2, targets_2 = imgs_2.to(self.device), targets_2.to(self.device)

                # Forward pass
                img_1_embedding, img_2_embedding = self.model(imgs_1, imgs_2)
                              
                # Create target vector by comparing the labels of the two images
                try:
                    targets = (targets_1 == targets_2).float() 
                except:
                    break      # if batches are different shapes that means we are at the 
                               # end of the iterator and can break out to the next epoch
                
                # Calculate loss and add to running total
                val_loss += self.criterion(img_1_embedding, img_2_embedding, targets).item()
                        
            val_loss /= len(self.val_loader)          # Average loss sum across the batch     
            self.historical_loss.append(val_loss)     # save loss for plotting
                
        return val_loss


