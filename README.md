# Siamese-Network Image Classification

## What is a Siamese Neural Network?

Siamse-Net is a neural network architecture that is designed to learn a mapping from
some input space to a smaller vector space such that similar inputs lead to vector 
embeddings in a similar location in the vector space. This means the network can be 
used to measure the similarity between two inputs, which is useful for tasks such as 
classification, search, and verification.

### Project Overview:
Originally, this project was intended for verification of handwritten signatures. However, I was
not able to source enough high quality data to train a model that I was truly satisfied
with. After coming back to the project after a few months, I'm going to re-purpose the 
project for image classification using the CIFAR-100 dataset, which contains 100 classes of images. 