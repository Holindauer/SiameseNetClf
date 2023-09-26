'''
    This script contains my modifications to the original siamese network model for the task of signature
    verification. TThe modifications to the model were done using the MNIST dataset for efficiency of 
    implementation. Below is a description of the current model architecture:

    The forward() function of the model recieves two seperate image tensors of the shape [batch, 1, height, width]
    In the case of mnist and the training params I used to develop this model, that would be [256, 1, 28, 28]

    These embeddings are passed into the encoder() method of the model. The encoder is a convolutional neural network
    that processes one of the images at a time. As the tensors move through the encoder, they reduce in dimmensionality.
    And before being output back into the forward pass again, they are flattened into a vector.

    The encoder uses 3 convolutional blocks. Each block consists of two convolutional layers, each followed by a batch
    normalization layer and a relu activation layer. Each block is followed by max pooling.

    Within the forward() pass, both of the images are passed through the encoder() method, resulting in two embeddings.
    These embeddings are each passed through a linear layer, which further reduces the size of the embedding to 10. 10
    was arbitrarily chosen during development.
    
    Both of these ten dimmensional vector embeddings are passed into the cosine_similarity() method, which computes the
    cosine similarity between the two vectors. The cosine similarity is defined as  A.dot(B) / |A| * |B| and is a measure
    of the similarity of the angle between the two vectors when normalized. The output of this method is a vector of size
    [batch]

    The cosine similarity vectors are then passed through the sigmoid function, which was chosen to replace torch.ge() because
    it is differentiable. torch.ge() not being differentiable was a key reason the original model was not converging. 

    These sigmoid outputs are the output of the model to which torch.ge() can be applied outside of the model to determine
    the accuracy of the model.

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 


class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        '''
            This method initializes the model. It takes a threshold value as input
            and stores it as an attribute. The threshold value is a hyperparameter
            that will be tuned. It also defines the convolutional blocks/layers
            that will be used in the encoder portion of the model.
        '''

        # define convolutional blocks/layers

        # 1 --> 4 channels
        self.conv_layer_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(4)  # Batch Norm for 4 output channels

        self.conv_layer_2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(4)  # Batch Norm for 4 output channels

        # 4 channels --> 8 channels
        self.conv_layer_3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.batch_norm_3 = nn.BatchNorm2d(8) # Batch Norm for 8 output channels

        self.conv_layer_4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.batch_norm_4 = nn.BatchNorm2d(8) # Batch Norm for 8 output channels

        # 8 channels --> 16 channels
        self.conv_layer_5 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.batch_norm_5 = nn.BatchNorm2d(16) # Batch Norm for 16 output channels

        self.conv_layer_6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.batch_norm_6 = nn.BatchNorm2d(16) # Batch Norm for 16 output channels


        # Dummy input to automatically determine the size of the linear layer
        dummy_input = torch.zeros(1, 1, 28, 28)  # Use batch size of 1 and same channels, height, width as actual input
        dummy_output = self.encoder(dummy_input)  # Pass dummy input through encoder method
        num_features = dummy_output.shape[1]      # Determine the number of features in the output

        # Define the linear layer
        self.linear_layer = nn.Linear(in_features=num_features, out_features=10)


    def encoder(self, input_tensor):
        '''
            This method defines the encoder portion of the model. It takes an image tensor
            as input and returns a vector embedding of the image. The encoder is a convolutional
            neural network. The input tensor is passed through the network, and the output is
            flattened and passed through a linear layer to reduce the size of the vector embedding.
        '''

        # conv block 1 ________1 --> 4 channels
        x = self.conv_layer_1(input_tensor)
        x = self.batch_norm_1(x)
        x = F.relu(x)

        x = self.conv_layer_2(x)
        x = self.batch_norm_2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # conv block 2 ________4 --> 8 channels
        x = self.conv_layer_3(x)
        x = self.batch_norm_3(x)
        x = F.relu(x)

        x = self.conv_layer_4(x)
        x = self.batch_norm_4(x)
        x = F.relu(x)

        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # conv block 3 ________8 --> 16 channels
        x = self.conv_layer_5(x)
        x = self.batch_norm_5(x)
        x = F.relu(x)

        x = self.conv_layer_6(x)
        x = self.batch_norm_6(x)
        x = F.relu(x)
        
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # compute number of features to flatten into from height and width
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


    def forward(self, images_1, images_2):
        '''
            This method defines the forward pass of the model. It takes
            a batch of image tensor pairs as input. It seperates the image
            pairs into two batched tensors and computes an embedding of each
            image in each batch. It then computes the cosine similarity between
            the two embeddings, and passes the result through a threshold function
            to return a 1 or 0 depending on whether the cosine similarity is greater
            than the threshold.
        '''

        # pass each image through the encoder
        embedding_1 = self.encoder(images_1)
        embedding_2 = self.encoder(images_2)

        # pass each embedding through the linear layer
        embedding_1 = self.post_encoder_linear(embedding_1)
        embedding_2 = self.post_encoder_linear(embedding_2)

        # due to dataset length differences, the last batch may be smaller than the others
        if embedding_1.size() != embedding_2.size():
            
            #if the last batch is smaller, return false
            return False

        # compute cosine similarity between the two embeddings
        cosine_similarity = self.cosine_similarity(embedding_1, embedding_2)

        # pass cosine similarity througj sigmoid function. Values in range [0, 1]
        # also, sigmoid is differentiable, so it can be used for backpropagation
        prediction = torch.sigmoid(cosine_similarity)

        return prediction

