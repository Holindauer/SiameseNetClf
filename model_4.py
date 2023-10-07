"""
    NOTE: Below is a note on the changes made recently to the model architecture:

    This model is the 4th redesign of the siamese network I have been developing 
    for signature verification. Convergence has been stalled for some time now 
    below 33% accuracy. 

    I was looking around at other siamese networks on the G-Hub and found this 
    implementation: https://github.com/seanbenhur/siamese_net/tree/master which 
    used a different method than I have been using for the loss function. It 
    appears to be a far superior method and makes my implementation look contrived. 

    The way I have implemented the siamese network + training up to now has been to 
    run a forward pass where I call the encoder component of the network twice, once
    for each image. Then flattening these outputs, and passing to a linear layer to 
    produce a vector embedding. These two vector embedding were then used to calculate
    the cosine similarity between them A.dot(B) / (||A|| * ||B||). The scalar value 
    returned was then passed through sigmoid activation and output from the model. 
    Because the model is outputing a single probability value, I have been using binary
    cross entropy as the loss function. 

    The implementation by seanbenhur uses a different approach. While the first part of 
    his implementation is the same as mine, that being to pass imgs through a CNN encoder,
    flatten, and pass through a linear layer to produce vector embeddings, he does not 
    compute the similarity function inside of the model. Instead, he uses the similarity 
    function as the loss funciton (and, interestingly, also to make the predictions from 
    the model output). The loss function he uses is euclidian distance. 

    This is a much better way to do it. The reason, I believe, my model kept failing was 
    because the flow of informative gradients was bottlenecked by the task being treated
    as a binary calssification problem. Since the task is to create a useful vector 
    embedding, being able to measure how close the vector embeddings are to each other
    should be what the loss informs about. The way I have been treating it up to now has 
    essentially been a very convoluded CNN classifier. 

    This implementation of the model only outputs 2 vector embeddings from the cnn encoder.
    The similarity function will be used to calculate the loss during training. It will be a
    modification of model 2. 

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 



class SiameseNet(nn.Module):
    def __init__(self, input_shape, embedding_size=128, initial_channel_increase=64, channel_increase_factor=2, dropout_rate=0.5):
        super().__init__()

        '''
            This class contains the architecture for the siamese network. The siamese network
            consists of 2 identical CNN encoders which share weights. Each encoder output is
            flattened and passed through a linear layer to produce a vector embedding, then 
            returned.
        '''

        # set dimmension of input image size
        self.input_shape = input_shape
        
        
        #_____________________ENCODER_____________________

        # -----Encoder Block 1-----

        # set channels for Block 1
        B1_channels = initial_channel_increase

        # conv layer 1
        self.conv_1 = nn.Conv2d(in_channels=input_shape[0], out_channels=B1_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(B1_channels)

        # conv layer 2
        self.conv_2 = nn.Conv2d(in_channels=B1_channels, out_channels=B1_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(B1_channels)

        # max pooling 
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)


        # -----Encoder Block 2-----

        # set channels for Block 2
        B2_channels = B1_channels * channel_increase_factor

        # conv layer 3
        self.conv_3 = nn.Conv2d(in_channels=B1_channels, out_channels=B2_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm_3 = nn.BatchNorm2d(B2_channels)

        # conv layer 4
        self.conv_4 = nn.Conv2d(in_channels=B2_channels, out_channels=B2_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm_4 = nn.BatchNorm2d(B2_channels)

        # max pooling 
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # -----Encoder Block 3-----

        # set channels for Block 3
        B3_channels = B2_channels * channel_increase_factor

        # conv layer 5
        self.conv_5 = nn.Conv2d(in_channels=B2_channels, out_channels=B3_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm_5 = nn.BatchNorm2d(B3_channels)

        # conv layer 6
        self.conv_6 = nn.Conv2d(in_channels=B3_channels, out_channels=B3_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm_6 = nn.BatchNorm2d(B3_channels)

        # max pooling
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        #_____________________CONV --> LINEAR_____________________

        # dropout layer to apply before linear to prevent overfitting
        self.dropout_1 = nn.Dropout(dropout_rate)

        # pass dummy input through encoder to get flattened encoder output shape
        dummy_input = torch.zeros(1, *input_shape)
        dummy_encoder_output = self.encoder(dummy_input)
        flattened_encoder_output_len = dummy_encoder_output.view(dummy_encoder_output.size(0), -1).shape[1]

        # use flattend encoder output len to  define linear 
        # layer to convert encoder output to latent dim
        self.post_encoder_linear = nn.Linear(in_features = flattened_encoder_output_len, out_features = embedding_size)




    def encoder(self, x):

        '''
            This function contains the forward pass of the encoder. The encoder consists
            of 3 convolutional blocks. Each block consists of 2 convolutional layers with
            batch normalization and ReLU activation. The encoder() method is called within
            the main forward() function of this class.
        '''

        # -----Encoder Block 1-----

        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        
        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = F.leaky_relu(x, negative_slope=0.01)

        x = self.max_pool_1(x)

        # -----Encoder Block 2-----

        x = self.conv_3(x)
        x = self.batch_norm_3(x)
        x = F.leaky_relu(x, negative_slope=0.01)

        x = self.conv_4(x)
        x = self.batch_norm_4(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        
        x = self.max_pool_2(x)


        # -----Encoder Block 3-----

        x = self.conv_5(x)
        x = self.batch_norm_5(x)
        x = F.leaky_relu(x, negative_slope=0.01)

        x = self.conv_6(x)
        x = self.batch_norm_6(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        
        x = self.max_pool_3(x)


        return x
    



    def forward(self, img_1, img_2):
        '''
            This function contains the forward pass of the autoencoder. The forward pass
            consists of passing the input through the encoder, flattening the output of
            the encoder, passing the flattened output through a linear layer to get the
            latent code, passing the latent code through a linear layer to get the
            pre-decoder output, reshaping the pre-decoder output to the shape of the
            encoder output before flattening, passing the reshaped output through the
            decoder, and padding the output of the decoder to counterbalance the loss of
            dimmensionality from the convolutional layers in the decoder.
        '''

        # _____________________ENCODER_________________________________

        img_1 = self.encoder(img_1)                  # pass both imgs through encoder
        img_2 = self.encoder(img_2)

        img_1 = img_1.view(img_1.size(0), -1)        # flatten output of encoder
        img_2 = img_2.view(img_2.size(0), -1)
        

        #_____________________LINEAR --> LATENT_____________________

        img_1 = self.dropout_1(img_1)               # apply dropout to prevent overfitting
        img_1 = self.post_encoder_linear(img_1)     # pass flattened output of encoder through

        img_2 = self.dropout_1(img_2)               # apply dropout to prevent overfitting
        img_2 = self.post_encoder_linear(img_2)     # pass flattened output of encoder through
        

        return img_1, img_2