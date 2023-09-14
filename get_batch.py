'''
The model being trained is a binary classifier that takes two image
tensors as input, maps each to a vector embedding, applies a similarity
function to them, and returns a 1 or 0 depending on whether the probability
returned by the similarity function is greater than a threshold.

This class will be used to build a batch of image pairs that will be fed into 
the model as input. The input to this class will be a stack of 2d image tensors 
The model will pair all of the images in the stack with each other, and return
a stack of image tensor pairs with the follwoing shape (batch_size, 2, height, width)

The 2 in the second dimmension being the two images in the pair

This class is intended to be applied after the class within image_collator has
already been applied to a list of PIL images.


TODO: I need to add labeling functionality to this class. 
'''
import torch

class Build_Batch:
    def __init__(self):
        pass

    def build_batch(self, tensor_stack):
        '''This method takes a stack of image tensors and returns a stack of image
        tensor pairs'''

        # instantiate a list to hold the pairs
        pair_list = []

        # iterate through the stack
        for i in range(tensor_stack.shape[0]):  # <---- example dimmension

            # iterate through the stack again
            for j in range(tensor_stack.shape[0]):

                # skip if the two images are the same
                if i == j:
                    continue

                else:
                    # stack two different tensors into a pair and append to list
                    pair_list.append(torch.stack([tensor_stack[i], tensor_stack[j]]))

        # stack the list of pairs and return
        return torch.stack(pair_list)