"""
@notice dataloader.py contains a custom data loader for use in the trianing of a siamese network. 

@dev What is this data loader solving? --- 
When training a siamese network for classification, you  need like class pairs to show up frequently in order 
to provide useful gradient information to the model regarding similar feature vectors during brackprop + sgd. 
However, if you just use a regular data loader configuration where the batch elements are more or less chosen 
at random, then you'll only get like class pairs passing into the model at the same frequency as exists for 
that class in the broader dataset. For example, in training a siamese net on the CIFAR-100, you will get roughly 
99 binary targets that are 0 and 1 that is 1 for a 100 element batch on average.

@dev This dataloader solves that problem by constructing a each batch of image pairs such that 50% are of the same 
class and the other 50 are chosen at random. These two groups are then combined in to a single Tensor and output 
for trainig on a single batch.

@dev NOTE: this dataloader is constructed specifically with CIFAR-100 in mind, as such the dataset is dowloaded 
from 

"""


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np

class SiameseClfDataloader(Dataset):
    def __init__(self, dataset: Dataset):
        """
        @param dataset (Dataset): The base dataset object (e.g., CIFAR-100 dataset).

        @dev The constructor stores the dataset and creates a mapping from labels to indices for quick lookup. 
        This is used to efficiently find other examples from the same class.
        """
        self.dataset = dataset
        self.labels = dataset.targets
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                 for label in np.unique(self.labels)}

    def __getitem__(self, index):
        """
        @dev Retrieve a pair of items from the dataset.

        Args:
            index (int): Index for the first image.

        @returns a tuple: (img1, img2, label) where img1 is the image at the provided index, img2 is a randomly 
        selected image (either from the same class or a different class), and label is 1 if both images are from 
        the same class, otherwise 0.
        """
        img1, label1 = self.dataset[index]

        # Randomly decide to create a positive or negative pair
        if np.random.randint(0, 2):
            # Positive pair
            siamese_index = index
            while siamese_index == index:
                siamese_index = np.random.choice(self.label_to_indices[label1])
        else:
            # Negative pair
            siamese_label = np.random.choice(list(set(self.labels) - set([label1])))
            siamese_index = np.random.choice(self.label_to_indices[siamese_label])
        
        img2, label2 = self.dataset[siamese_index]
        
        return img1, img2, torch.from_numpy(np.array([int(label1 == label2)], dtype=np.float32))

    def __len__(self):
        return len(self.dataset)

# Example of how to use this custom DataLoader
# cifar100_train = datasets.CIFAR100(root="data", train=True, download=True, transform=transforms.ToTensor())
# siamese_dataloader = DataLoader(SiameseClfDataloader(cifar100_train), batch_size=100, shuffle=True)
