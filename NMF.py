import numpy as np
from numpy import linalg as LA


class NMFDecomposition:
    def __init__(self, n_components, max_iter=500, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, return_loss=False):

        # Initialize W and H matrices randomly
        n_samples, n_features = X.shape

        avg = np.sqrt(X.mean() / self.n_components)
        rng = np.random.RandomState(seed=3)
        H = avg * rng.standard_normal(size=(self.n_components, n_features)).astype(
            X.dtype, copy=False
        )
        W = avg * rng.standard_normal(size=(n_samples, self.n_components)).astype(
            X.dtype, copy=False
        )
        H = np.abs(H, out=H)
        W = np.abs(W, out=W)

        # Perform NMF using multiplicative updates
        loss = []
        for i in range(self.max_iter):
            # Update H
            numerator = np.dot(W.T, X)
            denominator = np.dot(np.dot(W.T, W), H)
            H *= numerator / denominator

            # Update W
            numerator = np.dot(X, H.T)
            denominator = np.dot(W, np.dot(H, H.T))
            W *= numerator / denominator

            # Compute the error
            if return_loss:
                loss.append(LA.norm(X - W.dot(H), 'fro'))
            error = LA.norm(X - W.dot(H), 'fro')

            # Check for convergence
            if error < self.tol:
                break
        if return_loss:
            return W, H, loss
        return W, H

    def transform(self, X, H):
        n_samples = X.shape[0]

        avg = np.sqrt(X.mean() / self.n_components)
        rng = np.random.RandomState(seed=3)
        W = avg * rng.standard_normal(size=(n_samples, self.n_components)).astype(
            X.dtype, copy=False
        )
        W = np.abs(W, out=W)

        for i in range(self.max_iter):
            numerator = np.dot(X, H.T)
            denominator = np.dot(W, np.dot(H, H.T))
            W *= numerator / denominator

        return W
