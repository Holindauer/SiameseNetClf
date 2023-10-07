'''
    This file contains the class for a bi encoder model for representing images as 
    vector embeddings for the purpose of binary classification, similar in style 
    to one shot learning.

    Here is an outline of how the model is intended to work:

        For Context: I will briefly describe the format of the data. A batch of data 
                     contains a pair of (50, 150) images tensors. These tensors were 
                     augmented and collated by the class within image_collator.py. 
                     That class recieved a list of PIL images, applied .ToTensor()
                     (which normalized in addition to converting to tensor), then it 
                     applied maxpooling 2d a number of times specified before it is 
                     applied. The tensors are then resized with .Resize() to a standard
                     input size. Currently, I have set the input size to (50, 150). finally
                     they are stacked into a single tensor. 

        1.) The model takes as input a batch of image tensor pairs, to which each of them are 
            passed through the self.encoder() function. This function in a convolutional neural
            netword that returns a vector embedding of the image. Tensors of size (50, 150) when
            flattened, return a vector of size (1, 7500). 
            
            The goal is to bring this vector down to a size somewhere between (1, 128) and (1, 764)
            using convolution, then flatten it. I will need to experiment with the exact sizes that 
            will be used.

            Both images are passed through the encoder, resulting in two vector embeddings.  

        2.) Next, the cosine similarity between the two embeddings is calculated. This is done by
            taking the dot product of the two vectors, then dividing by the product of the two
            vector magnitudes. This results in a single value between 0 and 1.

            Cosine_Simiality = (A dot B) / (|A| * |B|)

            Similar to a logistic regression problem, this value will be treated as a probability of
            the two images being similar.

        3.) The cosine similarity value is then passed through a threshold function. This function
            will return a 1 if the value is greater than the threshold, and a 0 if it is less than
            the threshold. The threshold will be a hyperparameter that will be tuned.


    The model accepts a 5 dimensional tensor as input, and returns a 1 dimensional tensor as output.

    The input tensor has the following shape: (batch_size, 2, 1, 50, 150)

                                              (2, 1, 50, 150) refers to 2 images of shape (1, 50, 150)

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 


class BiEncoder(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        '''
            This method initializes the model. It takes a threshold value as input
            and stores it as an attribute. The threshold value is a hyperparameter
            that will be tuned. It also defines the convolutional blocks/layers
            that will be used in the encoder portion of the model.
        '''

        # define convolutional blocks/layers

        # 1 channel --> 8 channels
        self.conv_layer_1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv_layer_2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)

        # 8 channels --> 16 channels
        self.conv_layer_3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv_layer_4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)

        # 16 channels --> 32 channels
        self.conv_layer_5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_layer_6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)


        # Dummy input to automatically determine the size of the linear layer
        dummy_input = torch.zeros(1, 1, 50, 150)  # Use batch size of 1 and same channels, height, width as actual input
        dummy_output = self.encoder(dummy_input)  # Pass dummy input through encoder method
        num_features = dummy_output.shape[1]      # Determine the number of features in the output

        # Define the linear layer
        self.linear_layer = nn.Linear(in_features=num_features, out_features=128)

        # define threshold for positive/negative classification
        self.threshold_value = threshold


    def separate_images(self, input_tensor):
        """
            Separates an input tensor of shape [batch, 2, 1, 50, 150] into two tensors
            each of shape [batch, 1, 50, 150]. Used for separating the two images from
            the input tensor.
        """
        # Slice the tensor along the second dimension to separate the two images
        image1 = input_tensor[:, 0, :, :, :]  # shape will be [batch, 1, 1, 50, 150]
        image2 = input_tensor[:, 1, :, :, :]  # shape will be [batch, 1, 1, 50, 150]

        return image1, image2

    def encoder(self, input_tensor):
        '''
            This method defines the encoder portion of the model. It takes an image tensor
            as input and returns a vector embedding of the image. The encoder is a convolutional
            neural network. The input tensor is passed through the network, and the output is
            flattened and passed through a linear layer to reduce the size of the vector embedding.
        '''

        # conv block 1 ________1 --> 32 channels
        x = self.conv_layer_1(input_tensor)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # additional maxpooling layer from model_1.py
        x = self.conv_layer_2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # conv block 2 ________32 --> 64 channels
        x = self.conv_layer_3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # additional maxpooling layer form model_1.py
        x = self.conv_layer_4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # conv block 3 ________64 --> 128 channels
        x = self.conv_layer_5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # additional maxpooling layer from model_1.py
        x = self.conv_layer_6(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # compute number of features to flatten into
        num_features = x.size(1) * x.size(2) * x.size(3)

        # flatten output 
        return x.view(-1, num_features)


    def post_encoder_linear(self, input_tensor):
        '''
            This method defines the linear layer that is applied to the output of the encoder.
            It takes a tensor as input and computes a tensor of a smaller size. Then applies 
            tanh in the return statement.
        '''
        # compute linear layer
        linear_output = self.linear_layer(input_tensor)

        # apply tanh
        return torch.tanh(linear_output)

    def cosine_similarity(self, embedding_1, embedding_2):
        '''
            This method computes the cosine similarity between two vectors
            using the dot product and the magnitudes of the vectors. It is 
            written to work with batches data.

            The cosine similarity is defined by: A dot B / |A| * |B|
        '''
        # compute dot product
        dot_product = torch.einsum('ij,ij->i', embedding_1, embedding_2)
        
        # compute magnitudes
        mag_1 = torch.norm(embedding_1, dim=1)
        mag_2 = torch.norm(embedding_2, dim=1)

        # compute cosine similarity
        return dot_product / (mag_1 * mag_2)


    def threshold(self, pred):
        '''
            This method checks each value in a tensor and returns a 1 
            if the value is greater than the threshold, and a 0 if it 
            is less than the threshold. The threshold is a hyperparameter.
        '''
        # apply the greater than or equal to operator to each value in the tensor
        # this returns a boolena tensor to which we apply the float() function
        return torch.ge(pred, self.threshold_value).float()


    def forward(self, x):
        '''
            This method defines the forward pass of the model. It takes
            a batch of image tensor pairs as input. It seperates the image
            pairs into two batched tensors and computes an embedding of each
            image in each batch. It then computes the cosine similarity between
            the two embeddings, and passes the result through a threshold function
            to return a 1 or 0 depending on whether the cosine similarity is greater
            than the threshold.
        '''
        # separate the images
        image1, image2 = self.separate_images(x)

        # pass each image through the encoder
        embedding_1 = self.encoder(image1)
        embedding_2 = self.encoder(image2)

        # pass each embedding through the linear layer
        embedding_1 = self.post_encoder_linear(embedding_1)
        embedding_2 = self.post_encoder_linear(embedding_2)

        # compute cosine similarity between the two embeddings
        cosine_similarity = self.cosine_similarity(embedding_1, embedding_2)

        # pass cosine similarity through threshold function
        prediction = self.threshold(cosine_similarity)

        return prediction

