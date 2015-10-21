import numpy as np
from scipy.special import psi

def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(np.sum(alpha)))
    return(psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis])

class VB:
    """
    Compute perplexity, employing Variantional Bayes algorithm.
    """
    def __init__(self, _lambda, alpha, eta, max_iter):
        """
        Arguments:
            _Lambda: Variational parameters of topics of the learned model.
            alpha: Hyperparameter for prior on topic mixture theta.
            eta: Hyperparameter for prior on topics beta.
            max_infer: Number of iterations of FW algorithm.
        """
        self._lambda = np.copy(_lambda) + 1e-10
        self._K = _lambda.shape[0]
        self._W = _lambda.shape[1]
        self._alpha = alpha
        self._eta = eta
        self._max_iter = max_iter
        
        # normalize lambda
        _lambda_norm = self._lambda.sum(axis = 1)
        self._lambda /= _lambda_norm[:, np.newaxis]

    def do_e_step(self, batch_size, wordids, wordcts):
        """
        Does infernce for documents in 'w_obs' part.
        Arguments:
            batch_size: number of documents to be infered.
            wordids: A list whose each element is an array (terms), corresponding to a document.
                 Each element of the array is index of a unique term, which appears in the document,
                 in the vocabulary.
            wordcts: A list whose each element is an array (frequency), corresponding to a document.
                 Each element of the array says how many time the corresponding term in wordids appears
                 in the document.
        Returns: gamma the variational parameter of topic mixture (theta).
        """
        gamma = 1*np.random.gamma(100., 1./100., (batch_size, self._K))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)
        # Now, for each document d update that document's gamma and phi
        for d in range(0, batch_size):
            # Locate memory
            ids = wordids[d]
            cts = wordcts[d]
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self._lambda[:, ids]
            phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100
            # Iterate between gamma and phi until convergence
            for it in range(0, self._max_iter):
                gammad = self._alpha + expElogthetad * \
                    np.dot(cts / phinorm, expElogbetad.T)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = np.exp(Elogthetad)
                phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100
            gammad /= sum(gammad)
            gamma[d, :] = gammad
        return(gamma)
        
    def compute_lkh_d2(self, gammad, wordids_2d, wordcts_2d):
        """
        Compute log predictive probability for each document in 'w_ho' part.
        """
        ld2 = 0.
        frequency = 0
        for j in range(len(wordids_2d)):
            P = np.dot(gammad, self._lambda[:,wordids_2d[j]])
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
        gamma = self.do_e_step(batch_size, wordids_1, wordcts_1)
        # Compute perplexity
        LD2 = 0.
        for d in range(batch_size):
            LD2 += self.compute_lkh_d2(gamma[d], wordids_2[d], wordcts_2[d])
        return(LD2)    
