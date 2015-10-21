# -*- coding: utf-8 -*-

import numpy as np

class FW:
    """
    Compute perplexity, employing Frank-Wolfe algorithm.
    """
    
    def __init__(self, beta, max_iter):
        """
        Arguments:
            beta: Topics of the learned model.
            max_infer: Number of iterations of FW algorithm.
        """
        self.beta = np.copy(beta) + 1e-10
        self.num_topics = beta.shape[0]
        self.num_terms = beta.shape[1]
        self.INF_MAX_ITER = max_iter
        
        # Normalize beta
        beta_norm = self.beta.sum(axis = 1)
        self.beta /= beta_norm[:, np.newaxis]
        self.logbeta = np.log(self.beta)
        
        # Generate values used for initilization of topic mixture of each document
        self.theta_init = [1e-10] * self.num_topics
        self.theta_vert = 1. - 1e-10 * (self.num_topics - 1)
        
    def e_step(self, batch_size, wordids, wordcts):
        """
        Infer topic mixtures (theta) for all document in 'w_obs' part.
        """
        # Declare theta of minibatch
        theta = np.zeros((batch_size, self.num_topics))
        # Do inference for each document
        for d in range(batch_size):
            thetad = self.infer_doc(wordids[d], wordcts[d])
            theta[d,:] = thetad
        return(theta)
        
    def infer_doc(self, ids, cts):
        """
        Infer topic mixture (theta) for each document in 'w_obs' part.
        """
        # Locate cache memory
        beta = self.beta[:,ids]
        logbeta = self.logbeta[:,ids]
        # Initialize theta to be a vertex of unit simplex 
        # with the largest value of the objective function
        theta = np.array(self.theta_init)
        f = np.dot(logbeta, cts)
        index = np.argmax(f)
        theta[index] = self.theta_vert
        # x = sum_(k=2)^K theta_k * beta_{kj}
        x = np.copy(beta[index,:])
        # Loop
        for l in xrange(0,self.INF_MAX_ITER):
            # Select a vertex with the largest value of  
            # derivative of the objective function
            df = np.dot(beta, cts / x)
            index = np.argmax(df)
            beta_x = beta[index,:] - x
            alpha = 2. / (l + 3)
            # Update theta
            theta *= 1 - alpha
            theta[index] += alpha
            # Update x
            x += alpha * (beta_x)
        return(theta)
        
    def compute_lkh_d2(self, thetad, wordids_2d, wordcts_2d):
        """
        Compute log predictive probability for each document in 'w_ho' part.
        """
        ld2 = 0.
        frequency = 0
        for j in range(len(wordids_2d)):
            P = np.dot(thetad, self.beta[:,wordids_2d[j]])
            ld2 += wordcts_2d[j] * np.log(P)
            frequency += wordcts_2d[j]
        if frequency != 0:
            result = ld2 / frequency
        else:
            result = ld2
        return(result)
        
    def compute_perplexity(self, wordids_1, wordcts_1, wordids_2, wordcts_2):
        """
        Compute log predictive probability for all documents in 'w_ho' part.
        """
        batch_size = len(wordids_1)        
        # E step
        theta = self.e_step(batch_size, wordids_1, wordcts_1)
        # Compute perplexity
        LD2 = 0.
        for d in range(batch_size):
            LD2 += self.compute_lkh_d2(theta[d], wordids_2[d], wordcts_2[d])
        return(LD2)
