'''
    This script for the Contrastrive Loss funciton was found on github at the following link:
    https://github.com/seanbenhur/siamese_net/blob/master/siamese-net/contrastive.py

    It contains a class definition for the contrastic loss function for a siamese network

'''

import torch.nn as nn
from torch import Tensor
import numpy as np
import torch.nn.functional as F
import torch


class ContrastiveLoss(nn.Module):
    """
    @notice The contrastive loss function is a measure of the similarity of two vectors. Depending on a binary label, 
    of whether the vectors belong to the same class, the loss function will either penalize dissimilarity or
    similarity. The loss function is defined
    @dev The margin is a hyperparameter that defines the threshold at which the loss function will penalize dissimilarity 
    @dev The term (1-label) * torch.pow(euclidean_distance, 2) penalizes similarity when the vectors belong to different classs
    @dev The term (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2) penalizes dissimilarity when
    the vectors belong to the same class"""

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()

        self.margin = margin

    def forward(self, output1 : Tensor, output2 : Tensor, label : int):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + (label)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive