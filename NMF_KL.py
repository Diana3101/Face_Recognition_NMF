import numpy as np

EPSILON = np.finfo(np.float32).eps


class NMFDecomposition_KL:
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
        for _ in range(self.max_iter):

            # Update H
            X_WH = X / (W @ H)
            H *= W.T @ X_WH / np.sum(W, axis=0)[:, np.newaxis]

            # Update W
            X_WH = X / (W @ H)
            W *= X_WH @ H.T / np.sum(H, axis=1)[np.newaxis, :]

            # Compute the KL divergence
            indices = X > EPSILON
            WH_data = (W @ H)[indices]
            X_data = X[indices]

            # used to avoid division by zero
            WH_data[WH_data == 0] = EPSILON

            div = X_data / WH_data
            kl_div = np.dot(X_data, np.log(div))
            kl_div += np.sum(np.dot(W, H)) - X_data.sum()
            if return_loss:
                loss.append(kl_div)

            # Check for convergence
            if kl_div < self.tol:
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
            X_WH = X / (W @ H)
            W *= X_WH @ H.T / np.sum(H, axis=1)[np.newaxis, :]

        return W
