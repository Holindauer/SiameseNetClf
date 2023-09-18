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

I have written the torch code for data collation, batch building, and
the base model, which is a siamese network. I have also just completed
some data wrangling on the raw image data and am currently getting the 
training files ready for the first model.
