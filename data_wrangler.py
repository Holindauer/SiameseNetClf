"""
# Data Wrangling for Model 4 

From notebook --> .py source file

I began running into some issues during autoencoder training where the notebook I was using 
as a training env (which is an AWS sagemaker notebook w/ GPU) would suddenly run out of memory 
for tasks that seemingly should not take up so much memory. After looking into the differences 
between how jupyter vs native python deallocate unreference/derefrenced vars, I realize there 
was a major weakpoint in how I was preparing my data within these notebooks. I kept building on 
the data prep process with new functions. With each one, creating new variables for tensors with 
global scope to store the entire altered image dataset. This resulted in rapid consumption of memory. 

This script contains the same process from the data_wrangling_model_4.ipynb notebook, just moved 
into a python class. The reason for moving into a class is to move all temporary variables into 
the local scope of methods so they will be deallocate when the main() method is finished running.
"""

import os
from PIL import Image
import torchvision.transforms as transforms
import torch

class Wrangler:

    def __init__(self, op_sys, binary=False, reshape_size=(224, 224)):
        
        self.train_split=0.8
        
        if op_sys == "linux":
            self.path_seperator = "/"   
        elif op_sys == "windows":
            self.path_seperator = "\\"
        
        # make pixel intensties binary or not 
        self.binary = binary

        # reshape size for images
        self.reshape_size = reshape_size

    def main(self, train_dir, test_dir):   
        '''
            This is the main function of the class. It will call all other functions in the class
            to prepare the data for training. The only parameter is the path to the data directory
            that contains subdirs 'test' and 'train' with the image data. 
            
            The process is as follows:

            1. Get paths to class specific subdirs of images
            2. Load images and labels, applying transforms (making imgs binary)
            3. Concat train and test sets together for class balancing
            4. Get unique labels
            5. Group images by class into dictionary
            6. Split data into train and test sets with even class distributions
        '''
        
        # load paths to class specific subdirs of images
        train_paths = self.get_class_paths(train_dir)
        test_paths = self.get_class_paths(test_dir)

        # load images and labels, applying transforms
        train_labels, train_imgs = self.load_and_transform_imgs(train_paths)
        test_labels, test_imgs = self.load_and_transform_imgs(test_paths)

        # concat train and test sets together for class balancing
        images = torch.cat([test_imgs, train_imgs])
        labels = torch.cat([test_labels, train_labels])

        # get unique labels
        unique_labels = self.get_unique_labels(labels)

        # group images by class into dictionary
        label_img_dict = self.group_images_by_class(images, labels, unique_labels)

        # split data into train and test sets
        train_imgs, train_labels, test_imgs, test_labels = self.split_data(label_img_dict)

        return train_imgs, train_labels, test_imgs, test_labels


    def get_class_paths(self, data_dir):
        '''
            This method is used to gather all paths to subdirs of a given data directory.
            The subdirs are assumed to contain images of a single class. The method will
            also check if "forg" is in the subdir name. If so, it will not add the path
            to the list of paths. NOTE: This is specific to my current use case.
        '''

        #get list of subdirs, each containing a class of images
        dir_contents = os.listdir(data_dir)

        #create list of paths to each subdir
        paths_list = []

        # Get all the paths to the real signatures 
        for subdir in dir_contents:

            # only get paths to real signatures --- NOTE: This is specific to my current use case
            if "forg" not in subdir:

                # append full path to list
                paths_list.append(os.path.join(data_dir, subdir))

        return paths_list


    def load_and_transform_imgs(self, path_list):
        '''
            This method recieves a list of paths to directories containing multiple images, where the 
            name of the directory is the label for the images in the directory. The method iterates 
            through each path, converts the images to tensors and resizes to channelsx224x224 --> appends them 
            to a list. The method also creates a list of labels for each image in the path. These lists 
            are converted to tensors and stacked. The method returns a tensor of labels and a tensor of
            images.
        '''

        #--------------------------- open PIL --> apply transforms -----------------------

        # Initializes tranforms to apply to images
        tensor = transforms.ToTensor()
        resize = transforms.Resize(self.reshape_size)

        images, labels  = [], []   # initialize lists to store images and labels

        # iterate through each path in the list
        for path in path_list:

            # get label from path name (last element in path)
            label = int(path.split(self.path_seperator)[-1])

            # iterate through each img in the dir 
            for file in os.listdir(path):
                
                PIL_img = Image.open(os.path.join(path, file))      # open img from dir
                tensor_image = tensor(PIL_img)                      # PIL --> tensor
                tensor_image = resize(tensor_image)                 # resize --> 224x224

                images.append(tensor_image)                         # append tensor to list of images 

            dir_labels = [torch.tensor(label)] * len(os.listdir(path))   # list of tensor labels for each image in the dir 
            labels.append(torch.stack(dir_labels))                       # stack list of tensor labels into one tensor

        labels = torch.cat(labels)    #concat list of tensor stacks into one tensor
        images = torch.stack(images)  #stacked images into one tensor
        
        #---------------------------grayscale img conversion------------------------------
        
        gray_weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1) # tensor of ITU-R BT.601-7 weights with shape [1, 3, 1, 1] 
        images = (images * gray_weights).sum(dim=1, keepdim=True)           # broadcast gray weights across channels, 
                                                                            # sum channels --> single channel
            
        #---------------------------binary img conversion---------------------------------
        if self.binary:
            threshold = 0.9                                                 # threshold is 90% pixel intensity
            images = (images > threshold).float()                           # grayscale --> binary img w/ thresholding

        return labels, images

    def get_unique_labels(self, labels):
        '''
            This method iterates through the the labels 
            tensor and extracts all unique labels.
        '''
        unique_labels = []

        # iterate through number of examples to get unique labels
        for i in range(len(labels)):

            # get label of current iteration
            label = int(labels[i])

            # append label to list of unique labels if not already in list
            if label not in unique_labels:
                unique_labels.append(label)

        return unique_labels
    
    def group_images_by_class(self, images, labels, unique_labels):
        '''
            This method iterates through the the labels tensor and extracts all unique labels.
            These labels are then iterated through to extract out all images from the images tensor
            that share that label. The images are then returned in a dictionary with their label as key.
        '''

        # initialize dict to hold list of images for each unique label
        label_img_dict = {}
            
        for unique_label in unique_labels:   

            # intitialize list to hold images sharing a unique label
            like_images = []
            
            # iterate through labels tensor
            for (label, image) in zip(labels, images):

                # convert label to int
                label = int(label)

                # append only images that share the unique label
                if label == unique_label:
                    like_images.append(image)

            # add list of images to dictionary with their label as key
            label_img_dict[unique_label] = like_images

        return label_img_dict


    def split_data(self, label_img_dict):
        '''
            This method takes in a dictionary of labels and images and splits them into train and test sets.
            THe labels are used to balance the classes in the train and test sets. The method returns the
            train and test sets as tensors.
        '''

        # initialize lists to hold train and test images\labels
        train_images, train_labels = [], []
        test_images, test_labels = [], []

        # iterate through dictionary of labels and images
        for key, value in label_img_dict.items():

            # compute the train split index 
            train_split_index = int(len(value) * self.train_split)

            # extend train images and labels
            train_images.extend(value[:train_split_index])
            train_labels.extend([key] * train_split_index)  # set num indicies in train_split to key

            # extend test images and labels
            test_images.extend(value[train_split_index:]) 
            test_labels.extend([key] * (len(value) - train_split_index)) # set num indicies in test_split to key

        # stack train and test images and labels
        train_images = torch.stack(train_images)
        train_labels = torch.stack([torch.tensor(label) for label in train_labels])

        test_images = torch.stack(test_images)
        test_labels = torch.stack([torch.tensor(label) for label in test_labels])

        # return train and test images and labels
        return train_images, train_labels, test_images, test_labels
    
   