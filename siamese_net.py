import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List

@dataclass
class SiameseNetConfig:
    input_shape : tuple
    embedding_size : int
    initial_channels : int
    channel_increase_factor : int
    num_conv_blocks : int
    dropout_rate : float


class ConvBlock(nn.Module):
    """
    This class defines a convolutional block for use in the encoder of the siamese network. The conv block consists of
    two convolutional layers with batch normalization and leaky ReLU activation. 
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        # conv layer 1
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm_1 = nn.BatchNorm2d(out_channels)

        # conv layer 2
        self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)

        # max pooling 
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        
        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = F.leaky_relu(x, negative_slope=0.01)

        x = self.max_pool(x)

        return x
    

class SiameseNet(nn.Module):
    """ This class defines the siamese network. The siamese network consists of two encoders, each of which is a series of
    convolutional blocks. The output of the encoders is flattened and passed through a linear layer to get the vector
    embeddings of the input images."""
    def __init__(self, config: SiameseNetConfig):
        super().__init__()

        # unpack config
        input_shape = config.input_shape
        embedding_size = config.embedding_size
        initial_channels = config.initial_channels
        channel_increase_factor = config.channel_increase_factor
        num_conv_blocks = config.num_conv_blocks
        dropout_rate = config.dropout_rate


        # determine the number of channels for each convolutional block
        channels : List[int] = [initial_channels * (channel_increase_factor ** i) for i in range(num_conv_blocks)]

        # Initialize a ModuleList for Initializing ConvBlocks
        self.conv_blocks = nn.ModuleList()

        # Previous output channels, initialized to the input shape's channels
        prev_channels = input_shape[0]  # Assuming input_shape is (channels, height, width)

        # Instantiate ConvBlocks and append to the ModuleList
        for out_channels in channels:
            conv_block = ConvBlock(in_channels=prev_channels, 
                                   out_channels=out_channels,
                                   kernel_size=3,  # or any kernel size you want
                                   stride=1,
                                   padding=1)
            self.conv_blocks.append(conv_block)
            prev_channels = out_channels
            

        # dropout layer to apply before linear to prevent overfitting
        self.dropout = nn.Dropout(dropout_rate)

        # pass a dummy input through encoder to get flattened encoder output shape
        encoder_output = self.encoder(torch.zeros(1, *input_shape))
        output_elements = encoder_output.view(encoder_output.size(0), -1).shape[1] # get flattened encoder output len

        # use flattend encoder output len to define linear 
        self.linear = nn.Linear(in_features = output_elements, out_features = embedding_size)

    def encoder(self, x):
        ''' This function contains the forward pass of the encoder, consisting of passing the input through
        the convolutional blocks in the ModuleList, and returning the output of the final convolutional'''

        # Pass input through each ConvBlock
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        return x
    
    def forward(self, img_1, img_2):
        ''' This function contains the forward pass of the siamese network. It takes two images as input, passing each of
        them through the same encoder. The output of the encoder is then flattened and passed through a linear layer.'''

        # Encoder
        img_1 = self.encoder(img_1)        
        img_2 = self.encoder(img_2)
        
         # flatten both encoder outputs
        img_1 = img_1.view(img_1.size(0), -1)      
        img_2 = img_2.view(img_2.size(0), -1)

        # Apply droput regularization
        img_1 = self.dropout(img_1)   
        img_2 = self.dropout(img_2)            

        # pass through linear layer to get vector embeddings
        img_1 = self.linear(img_1)    
        img_2 = self.linear(img_2)     
        

        return img_1, img_2