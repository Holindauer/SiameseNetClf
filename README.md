# Signature-Similarity
Check Signature Similarity with Vector Embeddings

### Background into Project Inspo:
Lately, I have been captivated by the idea of models that can map an input 
to a vector space embedding. To which you can pass those embeddings into a
similarity functions to see how similar two inputs are semantically. 

I want to explore this idea further. My last project was a search engine that 
used this same method of vector embeddings and similarity functions to find lost 
files within a user inputted directory. In this project, I want to apply this 
same technique to image data. Specifically, to check the similarity of two signatures 
against each other.

Another important aspect of this project that I want to explore is to fully document
the entire process I went through the train each iteration of the model. I want to 
gain more insight into what model training strategies are best to go about first. 
As such, in the model_iterations directectory is the entire model training documentation,
containing my thoughts and analysis on each model and how to improve them. 

### Current Project Status:

Currently, I have attempted 3 different strategies for training the model, making slight
improvements to the validation accuracy in each one. 

I am currently building out model 4, which will use autoencoder pretraining as transfer 
learning, extracting out the encoder for use as the encoder of a siamese network model. 

### Accuracy Improvements 
Model 1 : < 0% validation accuracy
Model 2 : 25$ validation accuracy
Model 3 : 31.7% validation accuracy 


