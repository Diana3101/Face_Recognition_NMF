from dataloader import data_load
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from decompositions import decomposition_sklearn, decomposition
import pandas as pd
from sklearn.metrics import mean_squared_error

from classifier_models import classifier_short_version


def get_num_components(X_train, exp_threshold=0.9):
    pca = decomposition_sklearn(method_name="PCA", n_components=None)
    pca.fit(X_train)

    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)

    # Plot cumsum
    plt.plot(cumsum_variance)
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_ration')
    plt.show()
    return len(cumsum_variance[cumsum_variance < exp_threshold])


def visualize_nmf_decomposition(X, W, H, width, height):
    fig, axs = plt.subplots(nrows=1, ncols=9, figsize=(12, 2),
                            gridspec_kw={'width_ratios': [3, 0.001, 3, 0.001, 0.001, 3, 0.001, 3, 0.001]})
    axs[0].imshow((X[0] * 255).reshape((height, width)), cmap=plt.cm.gray)
    axs[0].axis('off')
    axs[1].text(0, 0.5, r'$\approx$', fontsize=20)
    axs[1].axis('off')
    axs[2].imshow((W.dot(H)[0] * 255).reshape((height, width)), cmap=plt.cm.gray)
    axs[2].axis('off')
    axs[3].text(0, 0.5, '=   ', fontsize=20)
    axs[3].axis('off')
    axs[4].text(0, 0.5, '$w_{i,1}$ *', fontsize=20)
    axs[4].axis('off')
    axs[5].imshow((H[0]).reshape((height, width)), cmap=plt.cm.gray)
    axs[5].axis('off')
    axs[6].text(0, 0.5, r'$w_{i,2}$ *', fontsize=20)
    axs[6].axis('off')
    axs[7].imshow((H[10]).reshape((height, width)), cmap=plt.cm.gray)
    axs[7].axis('off')
    axs[8].text(0, 0.5, r'$...$', fontsize=20)
    axs[8].axis('off')
    plt.show()


def plot_rmse(X_train):
    rmse = []
    for r in range(10, 300, 10):
        nmf = decomposition(method_name="NMF", n_components=r)
        W, H = nmf.fit(X_train)
        rmse.append(mean_squared_error(X_train, W.dot(H)))
    plt.plot(range(10, 300, 10), rmse)
    plt.xlabel('n_components')
    plt.ylabel('RMSE')
    plt.show()


def plot_losses(loss_nmf, loss_nmf_kl):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
    ax1.plot(loss_nmf)
    ax1.title.set_text('NMF (frobenius loss)')
    ax1.set_ylabel('Frobenius norm')
    ax2.plot(loss_nmf_kl)
    ax2.title.set_text('NMF (kullback-leibler loss)')
    ax2.set_ylabel('Kullback-Leibler divergence')
    fig.text(0.5, 0.04, 'Iterations', ha='center', fontsize=12)

    plt.show()


def test_classification():
    df_from_scratch = pd.DataFrame(index=['SVM', 'KNN', 'Random Forest'],
                      columns=['PCA', 'NMF (frobenius loss)',
                               'NMF (kullback-leibler loss)'])

    df_sklearn = pd.DataFrame(index=['SVM', 'KNN', 'Random Forest'],
                      columns=['PCA', 'NMF (frobenius loss)',
                               'NMF (kullback-leibler loss)'])
    X, y, w, h = data_load()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # print("Train:", X_train.shape)
    # print("Test:", X_test.shape)

    n_components = get_num_components(X_train)
    print('n_components: ', n_components)

    # plot_rmse(X_train)

    # PCA sklearn
    pca = decomposition_sklearn(method_name="PCA", n_components=n_components)
    pca.fit(X_train)
    X_train_decomposed = pca.transform(X_train)
    X_test_decomposed = pca.transform(X_test)

    df_sklearn['PCA'] = classifier_short_version(X_train_decomposed, y_train, X_test_decomposed, y_test)

    # PCA from scratch
    pca = decomposition(method_name="PCA", n_components=n_components)
    pca.fit(X_train)
    X_train_decomposed = pca.transform(X_train)
    X_test_decomposed = pca.transform(X_test)
    df_from_scratch['PCA'] = classifier_short_version(X_train_decomposed, y_train, X_test_decomposed, y_test)

    # NMF sklearn
    nmf = decomposition_sklearn(method_name="NMF", n_components=n_components)
    nmf.fit(X_train)
    X_train_decomposed = nmf.transform(X_train)
    X_test_decomposed = nmf.transform(X_test)

    df_sklearn['NMF (frobenius loss)'] = classifier_short_version(X_train_decomposed, y_train, X_test_decomposed, y_test)

    # NMF sklearn KL
    nmf = decomposition_sklearn(method_name="NMF_KL", n_components=n_components)
    nmf.fit(X_train)
    X_train_decomposed = nmf.transform(X_train)
    X_test_decomposed = nmf.transform(X_test)

    df_sklearn['NMF (kullback-leibler loss)'] = classifier_short_version(X_train_decomposed, y_train, X_test_decomposed, y_test)

    # NMF from scratch
    nmf = decomposition(method_name="NMF", n_components=n_components)
    X_train_decomposed, H, loss_nmf = nmf.fit(X_train, return_loss=True)
    X_test_decomposed = nmf.transform(X_test, H)
    visualize_nmf_decomposition(X=X_train, W=X_train_decomposed,
                                H=H, width=w, height=h)

    df_from_scratch['NMF (frobenius loss)'] = classifier_short_version(X_train_decomposed, y_train, X_test_decomposed, y_test)

    # NMF from scratch KL
    nmf = decomposition(method_name="NMF_KL", n_components=n_components)
    X_train_decomposed, H, loss_nmf_kl = nmf.fit(X_train, return_loss=True)
    X_test_decomposed = nmf.transform(X_test, H)

    df_from_scratch['NMF (kullback-leibler loss)'] = classifier_short_version(X_train_decomposed, y_train, X_test_decomposed, y_test)

    # Plot NMF and NMF_KL losses
    plot_losses(loss_nmf, loss_nmf_kl)

    df_from_scratch.to_csv('results_scratch.csv', index=False)
    df_sklearn.to_csv('results_sklearn.csv', index=False)


if __name__ == '__main__':
    test_classification()
