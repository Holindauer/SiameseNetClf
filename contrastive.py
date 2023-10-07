'''
    This script for the Contrastrive Loss funciton was found on github at the following link:
    https://github.com/seanbenhur/siamese_net/blob/master/siamese-net/contrastive.py

    It contains a class definition for the contrastic loss function for a siamese network

'''

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch


class ContrastiveLoss(nn.Module):
    "Contrastive loss function"

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + (label)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive