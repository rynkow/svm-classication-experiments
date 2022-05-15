import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
from scipy.spatial import distance_matrix


def contour_polt(xx, yy, ax, clf):
    predictions = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    predictions = predictions.reshape(xx.shape)
    ax.contourf(xx, yy, predictions, cmap="brg", alpha=0.3)


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # transform data with pca
    pca = PCA()
    X_transformed = pca.fit_transform(X)

    # Plot Covariance Matrices
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    mat = axs[0].matshow(np.cov(X, rowvar=False))
    plt.colorbar(mat, ax=axs[0])
    axs[0].set_title("Covariance matrix before PCA")
    mat = axs[1].matshow(np.cov(X_transformed, rowvar=False))
    plt.colorbar(mat, ax=axs[1])
    axs[1].set_title("Covariance matrix after PCA")
    fig.tight_layout()
    plt.savefig("iris/cov.png")
    plt.close('all')

    # reduce dimension to 2
    X_reduced = X_transformed[:, :2]

    # create grid to use in plots
    x_min, x_max = X_reduced[:, 0].min() - 0.2, X_reduced[:, 0].max() + 0.2
    y_min, y_max = X_reduced[:, 1].min() - 0.2, X_reduced[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # plot reduced data
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], s=8, c=y, cmap="brg")
    plt.title("Iris dataset after reduction to 2D")
    plt.savefig("iris/2dData.png")
    plt.close('all')

    # create subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # linear SVM
    linear_clf = SVC(kernel="linear")
    linear_clf.fit(X_reduced, y)

    # plot decision boundaries
    contour_polt(xx, yy, axs[0][0], linear_clf)
    axs[0][0].scatter(X_reduced[:, 0], X_reduced[:, 1], s=10, c=y, cmap="brg", edgecolors='gray')
    axs[0][0].label_outer()
    axs[0][0].set_title("linear SVM")

    # RBF kernel SVM
    rbf_clf = SVC(kernel="rbf", gamma="scale", C=2)
    rbf_clf.fit(X_reduced, y)

    # plot decision boundaries
    contour_polt(xx, yy, axs[0][1], rbf_clf)
    axs[0][1].scatter(X_reduced[:, 0], X_reduced[:, 1], s=10, c=y, cmap="brg", edgecolors='gray')
    axs[0][1].label_outer()
    axs[0][1].set_title("RBF kernel SVM")

    # Polynomial kernel SVM
    poly_clf = SVC(kernel="poly", gamma="scale", coef0=1, degree=3)
    poly_clf.fit(X_reduced, y)

    # plot decision boundaries
    contour_polt(xx, yy, axs[1][0], poly_clf)
    axs[1][0].scatter(X_reduced[:, 0], X_reduced[:, 1], s=10, c=y, cmap="brg", edgecolors='gray')
    axs[1][0].label_outer()
    axs[1][0].set_title("Polynomial kernel SVM")

    # custom kernel SVM
    def kernel(X, Y):
        R = 5
        distances = distance_matrix(X, Y)
        # multiply by matrix of booleans to filter values grater than R
        return (R - distances)/R*(distances <= R)
    custom_clf = SVC(kernel=kernel)
    custom_clf.fit(X_reduced, y)

    # plot decision boundaries
    contour_polt(xx, yy, axs[1][1], custom_clf)
    axs[1][1].scatter(X_reduced[:, 0], X_reduced[:, 1], s=10, c=y, cmap="brg", edgecolors='gray')
    axs[1][1].label_outer()
    axs[1][1].set_title("Custom kernel SVM")

    fig.suptitle("Iris dataset classification\n")
    fig.tight_layout()
    plt.savefig("iris/classification.png")







