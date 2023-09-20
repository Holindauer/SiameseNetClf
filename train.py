'''
This script contains the Trainer class, which is used to train the model.
It runs a basic pytorch training loop, with the option to use early stopping.
'''

import torch


class Trainer:
    def __init__(self, model, criterion, optimizer, train_loader, val_loader, epochs, device, early_stopping_patience, log_freq):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.early_stopping_patience = early_stopping_patience
        self.log_freq = log_freq

    def train(self):

        # initialize early stopping
        es_counter  = 0
        
        # training loop
        for epoch in range(self.epochs):
            
            # Train
            self.model.train()

            for batch_idx, (data, target) in enumerate(self.train_loader):

                # Load data to GPU
                data, target = data.to(self.device), target.to(self.device)

                # Clear gradients
                self.optimizer.zero_grad()

                # Forward pass
                output = self.model(data)

                # Calculate loss
                loss = self.criterion(output, target)

                # Backpropagation
                loss.backward()

                # Update weights
                self.optimizer.step()


            # Validate
            self.model.eval()

            val_accuracy = 0
            val_loss = 0
            with torch.no_grad():
                # Iterate over the validation data and accumulate the accuracy and loss.
                for data, target in self.val_loader:
                    
                    # Load data to device
                    data, target = data.to(self.device), target.to(self.device)

                    # Forward pass
                    output = self.model(data)

                    # Calculate loss and add to running total
                    val_loss += self.criterion(output, target).item()

                    # Calculate accuracy
                    pred = output.argmax(dim=1, keepdim=True)

                    # Add correct predictions to running total
                    val_accuracy += pred.eq(target.view_as(pred)).sum().item()

            # Average accuracy and loss across the batches.
            val_accuracy /= len(self.val_loader.dataset)
            val_loss /= len(self.val_loader.dataset)

            # Print metrics
            print(f'Epoch: {epoch}, Validation Accuracy: {val_accuracy}, Validation Loss: {val_loss}')



            # Check for early stopping
            if epoch == 0:
                best_val_accuracy = val_accuracy
                best_model = self.model
            else:
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_model = self.model
                    es_counter = 0
                else:
                    es_counter += 1

            if es_counter >= self.early_stopping_patience:
                print(f'Early stopping at epoch {epoch}')
                return best_model
            
            
