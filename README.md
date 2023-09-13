# Signature-Similarity
Check Signature Similarity with Vector Embeddings


Lately, I have been captivated by the idea of models that can map an input 
to a vector space embedding. To which you can pass those embeddings into a
similarity functions to see how similar two inputs are semantically. 

I want to explore this idea further. My last project was a search engine that 
used this same method of vector embeddings and similarity functions to find lost 
files within a user inputted directory. In this project, I want to apply this 
same technique to image data. Specifically, to check the similarity of two signatures 
against each other.

I plan to code up the model in pytorch and train it from scratch.  

Currently I am building out a dimmensionality reduction and collation pipeline 
for the image data. I want the pipeline to enable me to take photo of my signature 
with a scanner app on my phone and feed it into the model without any manual preperation.
