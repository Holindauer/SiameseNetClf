# Siamese-Network Image Classification

## What is a Siamese Neural Network?

Siamse-Net is a neural network architecture that is designed to learn a mapping from
some input space to a vector space. The network is designed to take in two inputs,
and output two vector embeddings. The network is trained to learn a mapping such that
the two vector embeddings are similar if the two inputs are similar, and dissimilar
if the two inputs are dissimilar.

This is done via comaparing the two vector embeddings using some similarity function
such as euclidian distance, cosine similarity, etc. The similarity function is used
to compute a similarity score, which is then compared to a similarity target. The
similarity target is 1 if the two inputs are similar, and 0 if the two inputs are
dissimilar. The similarity score is then compared to the similarity target using
some loss function such as binary cross entropy, contrastive loss, etc. The loss


### Project Overview:
Originally, this project was intended for verification of handwritten signatures. However, I was
not able to source enough high quality data to train a model that I was truly satisfied
with. After coming back to the project after a few months, I'm going to re-purpose the 
project for image classification using the CIFAR-100 dataset, which contains 100 classes of images. 