'''
    This script contains the third iteration of the model script. It defines a simaese network
    that uses Squeezenet 1.1, which was pretrained on ImageNet, as the encoder. The output of the 
    encoder is then passed through a linear layer to procduce a vector embedding of the image and 
    this embedding is then used to compute the cosine similarity between the two images. The 
    similarity is then passed through sigmoid and returned, to which a threshold function is applied 
    outside of the forward pass to determine whether the images are similar or not.

    The main difference between model 2 anbd model 3 is the addition of the use of transfer learning
    for the encoder. 

    There is also the addition of an additional similiarity function, euclidian distance, to which 
    the user can choose between it and cosine similarity during model instantiation. 

    

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torchvision.models as models
import torch.nn.functional as F



class SiameseNet(nn.Module):
    def __init__(self, input_shape, embedding_size, similarity_function):
        super().__init__()
        '''
            This method initializes the model. It takes a threshold value as input
            and stores it as an attribute. The threshold value is a hyperparameter
            that will be tuned. It also defines the convolutional blocks/layers
            that will be used in the encoder portion of the model.
        '''
        
        # Load pre-trained SqueezeNet 1.1
        # model outputs a 1000 dim vecotor (100 classes in ImageNet , which it was trained on)
        self.squeeze_net = models.squeezenet1_1(pretrained=True)

        # Define the linear layyer that follows sqqueeze net. Out size is the embedding size
        self.post_encoder_linear = nn.Linear(in_features=1000, out_features=embedding_size)
        
        # check which similarity_function was specified, and use to determine which method
        # to set self.similarity_func class variable to.
        
        if similarity_function == "cosine similarity":
            
            # set to cosine simialrity method
            self.similarity_function = self.cosine_similarity
            
            print(f"setting similarity function to {similarity_function}")
            
        elif similarity_function == "euclidian distance":
            
            # set to euclidian distance method
            self.similarity_function = self.euclidian_distance
        
            print(f"setting similarity function to {similarity_function}")
            
        else:
            print("error with similarity function")
            

    

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
    
    def euclidian_distance(self, embedding_1, embedding_2):
        '''
            This method provides an alternative similarity function
            to cosine similarity. That being euclidian distance.
        '''
        
        # F.pairwise_distance (euclidian sitance) expexts input tensors to be 2d
        embedding_1 = embedding_1.unsqueeze(0)
        embedding_2 = embedding_2.unsqueeze(0)
        
        return F.pairwise_distance(embedding_1, embedding_2).squeeze(0)


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
        embedding_1 = self.squeeze_net(images_1)
        embedding_2 = self.squeeze_net(images_2)

        # pass each embedding through the linear layer
        embedding_1 = self.post_encoder_linear(embedding_1)
        embedding_2 = self.post_encoder_linear(embedding_2)

        # due to dataset length differences, the last batch may be smaller than the others
        if embedding_1.size() != embedding_2.size():
            
            #if the last batch is smaller, return false
            return False
        
        # compute cosine similarity between the two embeddings
        similarity = self.similarity_function(embedding_1, embedding_2)

        # pass cosine similarity througj sigmoid function. Values in range [0, 1]
        # also, sigmoid is differentiable, so it can be used for backpropagation
        probability = torch.sigmoid(similarity)

        return probability

