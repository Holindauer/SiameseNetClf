# Siamese-Network Signature Verification

### Project Overview:
This project was to write a siamese network that creates vector embeddings 
for images of signatures that can be compared using some similarity function
for the task of signature verification. 

I have documented all model implementations that came before the current, most 
successful implementation. Historical implementations are stored within the 
historical-implementations subdirectory, which showcase data wrangling, training, 
previous architecutres, as well as my thought process along the way. A big part of 
this project was to understand better the ways in which I am not being efficient 
in setting out on my projects. This project is a significantly more technical 
challenge than all of my previous projects, mostly due to the higher dimmensionality
and complexity of the data I was working with, as well as the intricacies of the 
siamese network architecture and training process. 

### Pitfalls and Ladders
There were 4 model implementations in total. The first three showed mild improvements
in performance following each other. These improvements came from implementing standard
deep learning methodologies for better optimization. However, none of them performed well at all. This 
was because of a major logical error I had made in the initial 3 implementations. That 
being to use compute and threshold the similarity of the vector emebeddings within the 
forward pass, outputting a binary value of whether the two embeddings were similar or 
not. Thus, the model was trained using binary cross entropy as the criterion. This was
a massive error because it meant the model was not explicitly being trianed to create 
better mappings to the vector space, but instead, to perform binary vanilla classification
in a rather convoluded way. Although binary classification is the ultimate goal/utility
of the model, the actual training of the model needs to support the mechanism of which 
the architecture is designed to perform classification with (vector space mapping). 

What ended up getting the model to converge was to ditch Binary Cross Entropy and use the 
vector similarity function as the loss, which, in model 4, was contrastive loss (which uses
euclidian distance along with binary similarity targets). 

To answer the question I posed at the atart of this project of how I can be more efficient, 
I would say to be more intentional as to not attempt to optimize mechanisms that are useless. 
Avoid not doing the most obvious improvement.
