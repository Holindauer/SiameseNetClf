import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn
import copy
from dataclasses import dataclass
from early_stopping import EarlyStopping
from typing import Tuple

# training config
@dataclass
class TrainConfig:
    model: torch.nn
    trainloader: DataLoader
    testloader: DataLoader
    epochs: int
    device: torch.device
    contrastive_margin: float
    optimizer: torch.optim
    patience: int
    es_min_delta: float
    

class Trainer:
    def __init__(self, config: TrainConfig):   

        # unpack config
        self.model = config.model

        self.trainloader = config.trainloader
        assert self.trainloader.batch_size % 2 == 0, "Batch size must be even"

        self.testloader = config.testloader
        assert self.testloader.batch_size % 2 == 0, "Batch size must be even"

        self.epochs = config.epochs
        self.device = config.device
        self.optimizer = config.optimizer
        self.patience = config.patience

        # define loss function
        self.contrastive_loss = nn.CosineEmbeddingLoss(margin=0.5, reduction='mean')

        # training/test loss
        self.train_loss = [0 for i in range(self.epochs)]
        self.test_loss = [0 for i in range(self.epochs)]

        # setup early stopping
        self.early_stopping = EarlyStopping(patience=self.patience, min_delta=0.0)

    def train(self):
    
        #send model to device
        self.model.to(self.device)
        print(f"Model sent to {self.device}")

        for epoch in range(self.epochs):
            
            # set model to train mode
            self.model.train() 

     
            for (imgs1, imgs2, labels) in self.trainloader:

                #send data to device
                imgs1, imgs2 = imgs1.to(self.device), imgs1.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward 
                embedding1, embedding1 = self.model(imgs1, imgs2)

                # Compute the loss
                loss = self.contrastive_loss(embedding1, embedding1, labels)

                # Zero the gradients before running the backward pass
                self.optimizer.zero_grad()

                # Backward pass
                loss.backward()

                # Update weights
                self.optimizer.step()

                # update training loss
                self.train_loss[epoch] += loss.item()

            #average training loss
            self.train_loss[epoch] /= len(self.trainloader)

            # evaluate model on test set
            self.eval(epoch)

            # check for early stopping
            if self.early_stopping(self.test_loss[epoch], self.model):
                break

            print(f"Epoch {epoch+1} training loss: {self.train_loss[epoch]} test loss: {self.test_loss[epoch]}")

    def eval(self, epoch : int):

        # set model to eval mode
        self.model.eval()

        with torch.no_grad():
            for (imgs1, imgs2, labels) in self.testloader:
                    
                # send data to device
                imgs1, imgs2 = imgs1.to(self.device), imgs2.to(self.device)

                # forward 
                embedding1, embedding1 = self.model(imgs1, imgs2)

                # Compute the loss
                loss = self.contrastive_loss(embedding1, embedding1, labels)

                # Zero the gradients before running the backward pass
                self.optimizer.zero_grad()

                # Backward pass
                loss.backward()

                # Update weights
                self.optimizer.step()

                # update test loss
                self.test_loss[epoch] += loss.item()

            #average test loss
            self.test_loss[epoch] /= len(self.testloader)

            


                