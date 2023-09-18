"""
The class in this file is used to collate a list of PIL images into a single tensor.
It does this by applying max pooling and resizing to the images. The number of times
max pooling is applied is determined by the num_poolings parameter. The size to which
the images are resized is determined by the resize_size parameter. The print_shapes
parameter determines whether or not the shapes of the images are printed at each step

This collator was built with the intention to be applied before the trainign process.
"""

import torch, torchvision
import torch.nn as nn
from torchvision import transforms
import os

class ImageCollator:
    def __init__(self):
        pass

    def collate(self, list_of_images, num_poolings, print_shapes=False, resize_size=(50, 150)):
        ''' This is the main method of this class. It receives a list of PIL images
         and applies max pooling, grayscale, and resizing to them. It then returns a single tensor
          containing all of the images. The number of times max pooling is applied is
           determined by the num_poolings parameter. The size to which the images are
            resized is determined by the resize_size parameter. The print_shapes parameter
             determines whether or not the shapes of the images are printed at each step '''
        
        # convert list of images to grayscale
        list_of_images = self.PIL_to_grayscale(list_of_images, print_shapes)

        # convert list of images to list of tensors
        list_of_tensors = self.to_tensor(list_of_images, print_shapes)

        # apply max pooling to list of tensors
        pooled_tensors = self.pooling(list_of_tensors, num_poolings, print_shapes)

        # resize images to uniform size
        resized_tensors = self.resize(pooled_tensors, resize_size, print_shapes)

        # convert list of tensors to a single tensor and return
        return self.stack_tensors(resized_tensors, print_shapes)
    
    def PIL_to_grayscale(self, list_of_images, print_shapes=False):
        '''This method converts a list of PIL images to grayscale'''

        # instantiate transform
        grayscale_transform = transforms.Grayscale()

        # apply transform to each image and return
        grayscale_list = [grayscale_transform(image) for image in list_of_images]

        #print if desired
        if print_shapes:
            self.tensor_printer(grayscale_list, 'converting to grayscale')

        return grayscale_list


    def to_tensor(self, list_of_images, print_shapes=False):
        '''This method converts a list of PIL images to a list of tensors'''
        
        # instantiate transform
        tensor_transform = transforms.ToTensor()

        # apply transform to each image and return
        tensor_list = [tensor_transform(image) for image in list_of_images]
    
        #print if desired
        if print_shapes:
            self.tensor_printer(tensor_list, 'converting to tensor')

        return tensor_list



    def pooling(self, list_of_tensors, num_poolings, print_shapes=False):
        '''This method applies max pooling to a list of tensors. The number
        of times it is applied is determined by the num_poolings parameter'''

        # define a max pooling layer
        max_pool = nn.MaxPool2d(2, stride=2)

        # Apply max pooling to each tensor in the list num_poolings times
        for i in range(len(list_of_tensors)):
            for _ in range(num_poolings):
                list_of_tensors[i] = max_pool(list_of_tensors[i])

        #print if desired
        if print_shapes:
            self.tensor_printer(list_of_tensors, condition='max pooling')

        return list_of_tensors
    

    def resize(self, list_of_tensors, size, print_shapes=False):
        '''This method resizes a list of tensors to a uniform size'''

        # instantiate transform
        resize_transform = transforms.Resize(size)

        # apply transform to each tensor and return
        resized_tensors = [resize_transform(tensor) for tensor in list_of_tensors]

        #print if desired
        if print_shapes:
            self.tensor_printer(resized_tensors, condition='resizing')

        return resized_tensors

    
    def stack_tensors(self, list_of_tensors, print_shapes=False):
        '''This method stacks a list of tensors into a single tensor'''

        # stack tensors
        stacked_tensors =  torch.stack(list_of_tensors, dim=0)



        #print if desired
        if print_shapes:
            self.tensor_printer(stacked_tensors, condition='stacking')

        return stacked_tensors
    
    def tensor_printer(self, tensor_list, condition):
        '''This method prints the shapes of the tensors in a list of tensors
        It takes a list of tensors and a string condition to inform at what 
        point in the code the shapes are being printed'''

        print(f'\n\nShapes of images post {condition}: \n')

        # Check if tensor_list is actually a single tensor (stacked tensors)
        if isinstance(tensor_list, torch.Tensor):
            print(tensor_list.shape)
            return

        # Else assume it's a list of tensors
        for i, tensor in enumerate(tensor_list):
            print(tensor.shape, end=' ')
            if (i + 1) % 5 == 0:
                print() # Print a new line every 5 images

