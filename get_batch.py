'''
This class is used to build a batch of image tensor pairs. There are two methods that will do 
this in different ways. 

1.)
    The first method takes a stack of image tensors that are all of the same content. All possible 
    combinations of pairs will be made between the images in the stack. The stacks of pairs will
    then be stacked themselves into a single tensor. Additionally, they will be labeled as 1 to 
    represent that they the same image content. Such that the format is: 

    (batch_size, 2, 1, height, width, 1)  <----- this last 1 is the label (which is 1)

2.) 
    The second method will take a nested list of stacks of image tensors. These stacks, like the first
    method, are of the same content between the images in the stack. However, the different stacks are
    of different content. The method will build all possible combination of pairs between unlike images
    in the stack. The pairs stacks will be labeled with a zero to represent that they are not the same
    image content. Such that the format is:

    (batch_size, 2, 1, height, width, 1)  <----- this last 1 is the label  (which is 0)

'''
import torch

class Build_Batch:
    def __init__(self):
        pass

    def build_like_pairs(self, tensor_stack):
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

        # stack the list of pairs
        pair_stack = torch.stack(pair_list)

        # add a new dimension to the stack for the label
        pair_stack = pair_stack.unsqueeze(-1)

        # Fill the new dimension with ones to represent that the pairs are the same
        pair_stack[:, :, 0].fill_(1.0)

        return pair_stack

    
    def build_unlike_pairs(self, list_of_stacks):
        '''This method takes a list of stacks of image tensors and returns a stack of all possible 
        combinations of pairs of unlike images across all of the stacks'''

        #TODO: At current, this method does not check if there are duplicate pairs among the pair_list.

        # instantiate a list to hold the pairs
        pair_list = []

        # iterate through the list of stacks
        for i in range(len(list_of_stacks)):
            
            # iterate through all other stacks to create pairs across stacks
            for m in range(len(list_of_stacks)):

                # Skip pairing with the same so to access only images of unlike content
                if i == m:
                    continue

                # iterate through the i-th stack
                for j in range(list_of_stacks[i].shape[0]):

                    # iterate through the m-th stack
                    for k in range(list_of_stacks[m].shape[0]):

                        # stack two different tensors into a pair and append to list
                        pair_list.append(torch.stack([list_of_stacks[i][j], list_of_stacks[m][k]]))  

        # stack the list of pairs
        unlike_pairs_stack =  torch.stack(pair_list)

        # add a new dimension to the stack for the label
        unlike_pairs_stack = unlike_pairs_stack.unsqueeze(-1)

        # Fill the new dimension with zeros to represent that the pairs are not the same
        unlike_pairs_stack[:, :, 0].fill_(0.0)

        return unlike_pairs_stack

    
