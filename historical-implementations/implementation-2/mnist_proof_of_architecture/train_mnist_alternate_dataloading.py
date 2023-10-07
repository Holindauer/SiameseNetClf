'''
    NOTE: This script is a modified version of train_mnist.py. It utilizes a slightly different dataloading
          technique than what is described in train_mnist.py, mainly that the use of nested for loops that 
          each pull out a batch seperately has been removed and replaced with a system that uses python 
          iterators. This is described below in the third paragraph down. Other than that, the script and the
          description below are the same.

    This script is the modified training script for the siamese network model. Developed using the mnsit
    dataset, but with the intention of being used for signature verification. 

    The main goal of this training script adresses is to reduce the amount of data preperation that needs
    to be carried out before being passed into the model. The original model required the image data to be 
    a six dimmensional tensor containing all possible combinations of images grouped into pairs, to which
    the model would interanlly split into two seperate tensors and pass through the model. This has been 
    changed in this new implementation. Being replaced with a model that can take in two seperate tensors.
    
    -----> Here is the addition of the alternate dataloading technique <-----
    ---                                                                   ---
    The technique used within the training loop for loading in data is as follows: at the start of each epoch,
    a new python iterator is created from the torch dataloader. This iterator is then converted briefly into a 
    list to access the number of batches using len(). The list is also shuffled using random.shuffle(). After 
    these two operations, the suffled list is reverted back to an iterator. 
    
    Before the start of the epochs for loop (for running forward passes and backprop on each batch), half the 
    number of batches is computed. This is because the model requires two images per forward pass, so we must
    only iterate through half the num batches. Within this for loop described, the below operations take place:
    
    The two image batches are passed into the model which outputs a vector of size [batch] which contains the 
    cosine similarities of the two images after sigmoid has been applied ot them. To get the actual predictions 
    of the model, torch.ge() is applied to the output vector, which returns a vector of size [batch] containing
    binary values.

    In order to compute the loss, the targets of the two batches are created at each iteration of the inner loop 
    by comparing the class to which each image fo the batch belongs. This resultant target vector and the model
    prediction vector are then passed into the criterion (BCE) and backrpopagation is carried out.

    The validation method applies a different mechanism for creating pairs of images. The reason for this is to 
    avoid the issue of having an extremely large number of pairs of images to compare. The training loop soves this 
    with the epoch check within the outer loop. Whcih will break training once the set number of epochs has been
    exceeded. The validation set however, is not subject to epochs. The solution I employed for the val loop was to 
    create a python iterator from the val loader and pull out two images per iteration. 

    There is also early stopping regularization that is performed after each epoch. 
'''

import torch
import copy
import random


class Trainer:
    def __init__(self, model, criterion, optimizer, train_loader, val_loader, epochs, device, early_stopping_patience, log_freq, batch_size):   
        # model parameters
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device

        # data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader         
        
        # logging parameters
        self.log_freq = log_freq  
        self.batch_size = batch_size
        self.historical_loss = []
        self.historical_acc = []

        # early stopping parameters
        self.early_stopping_patience = early_stopping_patience
        self.es_counter = 0
        self.best_val_accuracy = 0
        self.best_model = None

    def train(self):
    
        #send model to device
        self.model.to(self.device)

        # ensure model is in train mode
        self.model.train()

        # training loop
        for epoch in range(self.epochs):

            # initialize a sum for train loss and accuracy for the epoch
            train_loss, train_accuracy = 0, 0

            # reset train iterator
            train_iterator = iter(self.train_loader)
            
            # convert iterator to list so random.shuffle() can be applied
            train_iterator_list = list(train_iterator)
            random.shuffle(train_iterator_list)
            
            # convert iterator list back to iterator
            train_iterator = iter(train_iterator_list)
            
            # compute half of the num batches for the training loop
            half_of_num_batches = int(len(train_iterator_list) / 2)
            
            for i in range(half_of_num_batches): #    <--- Iterate through each 1/2 the batches of the 
                                                 #         trainloader since 2 batches pulled per inference
                
                #protect against non-even batch causing Stop-Iteration Error
                try:
                    #get the next two batchs of the iterator
                    imgs_1, targets_1 = next(train_iterator)
                    imgs_2, targets_2 = next(train_iterator)
                # if there are is only one batch left, that means we are at the end
                # of the iterator and can break out to the next epoch
                except:
                    break  
                    
                
                # Load images to GPU
                imgs_1, targets_1 = imgs_1.to(self.device), targets_1.to(self.device)
                imgs_2, targets_2 = imgs_2.to(self.device), targets_2.to(self.device)
                
                # Clear gradients
                self.optimizer.zero_grad()

                # Forward pass -- output is a vector of cosine similarities with sigmoid applied
                output = self.model(imgs_1, imgs_2)

                # The last batch may be smaller than the others. If this is the case, the model will return False
                # in this case, which will break to the next batch of the outer loop, starting the next epoch.
                if not isinstance(output, torch.Tensor):
                    break

                # create target vector by comparing the labels of the two images
                targets = (targets_1 == targets_2).float()

                # send target to device
                targets = targets.to(self.device)

                # Calculate loss and add to running total
                loss = self.criterion(output, targets)
                train_loss += loss.item()

                # compute step accuracy and add to running total for epoch
                # threshold function applied to raw probability output to compute predictions
                predictions = torch.ge(output, 0.5)
                train_accuracy += (predictions == targets).float().sum().item()
                   
                # Backpropagation
                loss.backward()

                # Update weights
                self.optimizer.step()
                
            # Average train loss sum across half the num batches (2 imgs per example) 
            train_loss /= half_of_num_batches

            # Average number of correct predictions across number fo examples in the half num examples 
            train_accuracy /= half_of_num_batches * self.batch_size
                
            # Eval on the validation set
            val_accuracy, val_loss = self.validate()
           
            # print epoch performance
            print(f"Epoch {epoch}: Train Loss: {train_loss:.5f} -- Train Accuracy {train_accuracy:.3f} -- Val Loss: {val_loss:.5f} -- Val Accuracy: {val_accuracy:.3f}")
  
            # Check for early stopping -- es_counter class var incremented in early_stopping method
            self.early_stopping(epoch, val_accuracy)

            # determine if early stopping patience has been exceeded, stop training if so
            if self.es_counter >= self.early_stopping_patience:
                break
          
        return self.best_model, self.historical_loss, self.historical_acc

            
    def validate(self):
            
        # Validate
        self.model.eval()

        # initialize a sum for val loss and accuracy for the epoch
        val_loss, val_accuracy = 0, 0
        
        with torch.no_grad():
            
            # reset val iterator
            val_iterator = iter(self.val_loader)

            # iterate over the validation set
            for i in range(len(val_iterator)):

                # get the next batch of images and targets from the validation set
                imgs_1, targets_1 = next(val_iterator)
                imgs_2, targets_2 = next(val_iterator)

                # Load images to GPU
                imgs_1, targets_1 = imgs_1.to(self.device), targets_1.to(self.device)
                imgs_2, targets_2 = imgs_2.to(self.device), targets_2.to(self.device)

                # Forward pass
                output = self.model(imgs_1, imgs_2)
                        
                # due to dataset length differences, the last batch may be smaller than the others
                # if this is the case model will return false, which will break out to the next batch 
                # of the outer loop.
                if not isinstance(output, torch.Tensor):
                    break
                        
                            
                # Create target vector by comparing the labels of the two images
                targets = (targets_1 == targets_2).float()

                # Calculate loss and add to running total
                val_loss += self.criterion(output, targets).item()
                        
                # compute step accuracy and add to running total for the val set
                # threshold function applied to raw output to compute predictions
                predictions = torch.ge(output, 0.5)
                val_accuracy += (predictions == targets).float().sum().item()

            # Average loss and accuracy sums across the batch
            val_accuracy /= len(self.val_loader.dataset)
            val_loss /= len(self.val_loader)     
                    
            #save loss and acc for plotting
            self.historical_loss.append(val_loss)
            self.historical_acc.append(val_accuracy)    
                
        return val_accuracy, val_loss


    def early_stopping(self, epoch, val_accuracy):

        # Check for early stopping
        if epoch == 0:

            #set intialize best acc and model for epoch 0
            self.best_val_accuracy = val_accuracy 
            self.best_model = self.model
            
        else: #otherwise check if accuracy has improved
            
            if val_accuracy > self.best_val_accuracy:
   
                #save best model and its performance and reset es_counter
                self.best_val_accuracy = val_accuracy
                self.best_model = copy.deepcopy(self.model)
                self.es_counter = 0
                
            else:
                #otherwise add 1 to counter
                self.es_counter += 1
