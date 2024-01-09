import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn
from dataclasses import dataclass
from early_stopping import EarlyStopping

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
        # Unpack config
        self.model = config.model
        self.trainloader = config.trainloader
        assert self.trainloader.batch_size % 2 == 0, "Batch size must be even"
        self.testloader = config.testloader
        assert self.testloader.batch_size % 2 == 0, "Batch size must be even"
        self.epochs = config.epochs
        self.device = config.device
        self.optimizer = config.optimizer
        self.patience = config.patience

        # Define loss function
        self.contrastive_loss = nn.CosineEmbeddingLoss(margin=config.contrastive_margin, reduction='mean')

        # Training/test loss and accuracy
        self.train_loss = [0 for i in range(self.epochs)]
        self.test_loss = [0 for i in range(self.epochs)]
        self.train_accuracy = [0 for i in range(self.epochs)]
        self.test_accuracy = [0 for i in range(self.epochs)]

        # Setup early stopping
        self.early_stopping = EarlyStopping(patience=self.patience, min_delta=0.0)

    def train(self):
        
        self.model.to(self.device)
        print(f"Model sent to {self.device}")

        for epoch in range(self.epochs):
            
            self.model.train() 
            correct_train_predictions = 0
            total_train_predictions = 0

            for (imgs1, imgs2, labels) in self.trainloader:
                imgs1, imgs2 = imgs1.to(self.device), imgs2.to(self.device)
                labels = labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward 
                embedding1, embedding2 = self.model(imgs1, imgs2)
                labels = labels.squeeze()
                loss = self.contrastive_loss(embedding1, embedding2, labels)

                # Backward pass and update weights
                loss.backward()
                self.optimizer.step()

                # Update training loss
                self.train_loss[epoch] += loss.item()

                # Calculate predictions and update accuracy
                cos_sim = torch.nn.functional.cosine_similarity(embedding1, embedding2)
                predicted_labels = (cos_sim > 0.5).float()
                correct_train_predictions += (predicted_labels == labels).sum().item()
                total_train_predictions += labels.size(0)

            # Average training loss and accuracy
            self.train_loss[epoch] /= len(self.trainloader)
            self.train_accuracy[epoch] = correct_train_predictions / total_train_predictions

            # Evaluate model on test set
            self.eval(epoch)

            # Check for early stopping
            if self.early_stopping(self.test_loss[epoch], self.model):
                break

            print(f"Epoch {epoch+1}: Training Loss: {self.train_loss[epoch]}, Training Accuracy: {self.train_accuracy[epoch]}, Test Loss: {self.test_loss[epoch]}, Test Accuracy: {self.test_accuracy[epoch]}")

    def eval(self, epoch: int):
        self.model.eval()
        correct_test_predictions = 0
        total_test_predictions = 0

        with torch.no_grad():
            for (imgs1, imgs2, labels) in self.testloader:
                imgs1, imgs2 = imgs1.to(self.device), imgs2.to(self.device)
                labels = labels.to(self.device)

                # Forward
                embedding1, embedding2 = self.model(imgs1, imgs2)
                labels = labels.squeeze()
                loss = self.contrastive_loss(embedding1, embedding2, labels)

                # Update test loss
                self.test_loss[epoch] += loss.item()

                # Calculate predictions and update accuracy
                cos_sim = torch.nn.functional.cosine_similarity(embedding1, embedding2)
                predicted_labels = (cos_sim > 0.5).float()
                correct_test_predictions += (predicted_labels == labels).sum().item()
                total_test_predictions += labels.size(0)

            # Average test loss and accuracy
            self.test_loss[epoch] /= len(self.testloader)
            self.test_accuracy[epoch] = correct_test_predictions / total_test_predictions
