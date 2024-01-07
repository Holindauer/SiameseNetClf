import torch
from torch import Tensor
from torch.utils.data import DataLoader
import copy
from dataclasses import dataclass
from contrastive import ContrastiveLoss
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
        self.contrastive_loss = ContrastiveLoss(margin=2.0)  

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

     
            for (batch, label) in self.trainloader:
                
                # split batch in half
                data1, data2, binary_labels = self.split_batch(batch, label)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward 
                embedding1, embedding1 = self.model(data1, data2)

                # Compute the loss
                loss = self.contrastive_loss(embedding1, embedding1, binary_labels)

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
            for (batch, label) in self.testloader:
                    
                # split batch in half   
                data1, data2, binary_labels = self.split_batch(batch, label)

                # forward 
                embedding1, embedding1 = self.model(data1, data2)

                # Compute the loss
                loss = self.contrastive_loss(embedding1, embedding1, binary_labels)

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

            
    def split_batch(self, batch : Tensor, label : Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        @notice This function splits a batch of data into two batches of equal size. The labels are also split into two batches
        @dev This is because the siamese network takes two images as input
        """

        # send data to device
        batch : Tensor = batch.to(self.device)
        label = label.to(self.device)

        # split data into two batches
        split : int = batch.shape[0]//2
        data1 : Tensor = batch[:split]
        data2 : Tensor = batch[split:]

        # split labels into two batches
        label1 : Tensor = label[:split]
        label2 : Tensor = label[split:]
                    
        # use boolean masking to determine like images
        binary_labels : Tensor = (label1 == label2).float()

        return data1, data2, binary_labels

                