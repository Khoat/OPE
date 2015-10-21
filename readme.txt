***********************************************************************
Posterior inference in topic models with provably guaranteed algorithms
***********************************************************************

(C) Copyright 2015, Khoat Than and Tung Doan. 
This package is for academic uses only. Other usages should ask for permission.

------------------------------------------------------------------------
This Python package contains six algorithms for learning Latent Dirichlet 
Allocation (LDA) at large scales, including: ML-FW, Online-FW, Streaming-FW, ML-OPE, Online-OPE, Streaming-OPE. They are stochastic algorithms that can work with big text collections and text streams. Their cores are two fast inference approaches to understanding individual texts.

The two inference approaches are Frank-Wolfe (FW) and Online Maximum a Posterior Estimation (OPE). FW has linear convergence rate, offers a principled way to trade off sparsity of solutions against quality. OPE theoretically converges to local optimum
or stationary point with a linear rate. These two methods can be easily employed to do posterior inference for various probabilistic models.

If you find the package useful, please consider to cite our related work:

Khoat Than and Tu Bao Ho. Inference in topic models I: sparsity trade-off. Technical report, 2015.
Khoat Than and Tung Doan. Inference in topic models II: provably guaranteed algorithms. Technical report, 2015.

------------------------------------------------------------------------
TABLE OF CONTENTS


A. LEARNING ALGORITHMS

   1. SETTINGS FILE

   2. DATA FILE FORMAT

B. MEASURE

C. PRINTING TOPICS


------------------------------------------------------------------------
A. LEARNING ALGORITHMS

Implementations are included in folders each of which is 
named according to the name of the corresponding algorithm.
Each folder includes a file implementing the algorithm and 
the other file run the algorithm to learn model from a large
corpus.

Estimate a model by executing:

     python run_[name of algorithm].py  [train file] [setting file] 
[model folder] [test data folder]

[train file]                      path of the training data.
[setting file]                    path of setting file provides parameters 
                                  for learning.
[model folder]                    path of the folder for saving the learned model.
[test data folder]             	  path of the folder contains data for computing
                                  perplexity (described in details in B).

The model folder will contain some more files. These files contain some statistics of how the model is after a mini-batch is processed. These statistics include topic mixture sparsity, perplexity of the model, top ten words of each topic, and time for finishing the E and M steps. 

Example: python ./run_ML_FW.py ../data/nyt_50k.txt ../settings.txt ../models/ML_FW/nyt ../data

1. Settings file

See settings.txt for a sample.

2. Data format

The implementations only support reading data type in LDA. Please refer to the following site for instructions.

http://www.cs.columbia.edu/~blei/lda-c/

Under LDA, the words of each document are assumed exchangeable.  Thus, each document is succinctly represented as a sparse vector of word counts. The data is a file where each line is of the form:

     [M] [term_1]:[count] [term_2]:[count] ...  [term_N]:[count]

where [M] is the number of unique terms in the document, and the [count] associated with each term is how many times that term appeared in the document.  Note that [term_1] is an integer which indexes the term; it is not a string.


------------------------------------------------------------------------

B. MEASURE

Perplexity is a popular measure to see predictiveness and generalization of a topic model.

In order to compute perplexity of the model, the testing data is needed. Each document in testing data is randomly divided into two disjoint part w_obs and w_ho. After 5 times of independent dividing the testing data, there are 5 data couples (w_obs, w_ho)
They are stored in [divided data folder] with corresponding file name is of the form:

data_test_[i]_part_1.txt and data_test_[i]_part_2.txt

where [i] is in range 1 to 5, specifying the ordinal number of the data couple. Perplexity is computed from all 5 data couples after processing a mini-batch.


------------------------------------------------------------------------

D. PRINTING TOPICS

The Python script topics.py lets you print out the top N
words from each topic in a .topic file.  Usage is:

     python topics.py [beta file] [vocab file] [n words] [result file]
