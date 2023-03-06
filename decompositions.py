from sklearn.decomposition import PCA, NMF
from NMF import NMFDecomposition
from PCA import PCADecomposition
from NMF_KL import NMFDecomposition_KL


def decomposition_sklearn(method_name="PCA", n_components=10):
    clf = None
    if method_name == "PCA":
        clf = PCA(n_components=n_components)
    elif method_name == "NMF":
        clf = NMF(n_components=n_components, init='random', solver='mu',
                  beta_loss='frobenius', random_state=3, max_iter=500)
    elif method_name == "NMF_KL":
        clf = NMF(n_components=n_components, init='random', solver='mu',
                  beta_loss='kullback-leibler', random_state=3, max_iter=500)

    return clf


def decomposition(method_name="PCA", n_components=10):
    clf = None
    if method_name == "PCA":
        clf = PCADecomposition(n_components=n_components)
    elif method_name == "NMF":
        clf = NMFDecomposition(n_components=n_components)
    elif method_name == "NMF_KL":
        clf = NMFDecomposition_KL(n_components=n_components)

    return clf
