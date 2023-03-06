import numpy as np


class PCADecomposition:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Calculate the covariance matrix
        cov = np.cov(X.T)

        # Get the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort the eigenvectors in descending order of eigenvalues
        indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, indices]

        # Get the top n_components eigenvectors
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        # Center the data
        X = X - self.mean

        # Project the data onto the top n_components eigenvectors
        return np.dot(X, self.components)
