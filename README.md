# Siamese-Network Image Classification

This repository implements a Siamese Neural Network for the task of multi-class image classificaiton on the CIFDAR-100 dataset.

## Siamese Net Architecture

The siamese net is a neural network architecture that learns a mapping from some input space to a smaller embedding space. The network accepts two (potentially more) labeled inputs, passing each of them 
through the same network. That is, two separate forward passes are performed using the same weights for each input. The network outputs two vector embeddings, which can be compared using a distance metric for tasks such as classification, search, and verification. 

The benefit of this approach is that the network can precompute embeddings and store them in a database, which can be queried at a later time. For the application of search, storing precomputed embeddings for similarity comparison is vastly more efficient than, say, running inference on a binary classifier optimized to determine if a query matches an input for the entire database.

## Training 

The siamese network is optimized to minimize a distance metric for inputs of the same class, and maximize that distance for inputs of different classes. In this implementation I chose the 
contrastive loss function, which is defined as follows:

$$
J(\theta) = \frac{1}{2} \left[ y \cdot D^2 + (1 - y) \cdot \max(0, m - D)^2 \right]
$$

Where:
- $D$ is the distance metric between embeddings 
- $y$ is a binary label regarding if the inputs belong to the same class
- $m$ is the margin, a hyperparameter that controls the minimum allowed distance between embeddings for inputs of different classes.

This loss function can be understood as the average of the two inner terms. When $y$ is 1, the first term is used, and the network is rewarded for minimizing the distance between embeddings. When $y$ is 0, the second term is used, and the network is penalized for embeddings that are closer than the margin $m$.

I conducted training in an AWS ml.g4dn.xlarge sagemaker notebook instance, all model/training components being imported into this [notebook](training_env.ipynb). The training script is located in [train.py](train.py). The model was written in pytorch, and can be found in [siamese_net.py](siamese_net.py). As well, I am using a custom [dataloader](dataloader.py) to ensure equeal class distribution in each batch. The distribution of like and unlike pairs that is, not the distribution of classes. As well as [early stopping regularization](early_stopping.py).



### Project History:
Originally, this project was intended for verification of handwritten signatures. However, I was
not able to source enough high quality data to train a model that I was truly satisfied
with. After coming back to the project after a few months, I'm going to re-purpose the 
project for image classification using the CIFAR-100 dataset, which contains 100 classes of images. 