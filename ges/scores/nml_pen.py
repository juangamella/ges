import numpy as np
from .decomposable_score import DecomposableScore

class NmlPen(DecomposableScore):
    def __init__(self, data, cache=True, debug=0):
        """ Cretes a new instance of the class.      

        Parameters
        ----------

        """

        super().__init__(data, cache=cache, degug=debug)

        self.n, self.p = data.shape

    def full_score(self, A):
        """ Given a DAG adjacency A, return nml-penalized log-likelihood of a sample
        """
        pass

    def _compute_local_score(self, x, pa):
        """ Given a node and its parents, return nml-penalized log-likelihood
        of a sample from a single environment
        """
        pass

    def _mle_full(self, A):
        pass

    def _mle_local(self, A):
        pass


        
