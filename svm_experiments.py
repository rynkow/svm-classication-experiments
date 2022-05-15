import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
from scipy.spatial import distance_matrix

DATA_PICTURE = "svm_data.png"


def load_data_from_picture(picture):
    # make picture grayscale
    picture = cv.cvtColor(picture, cv.COLOR_BGR2GRAY)

    # load data - white pixels and black pixels represent 2 classes
    data = []
    labels = []
    for i in range(picture.shape[0]):
        for j in range(picture.shape[1]):
            if picture[i][j] == 0:
                data.append([i, j])
                labels.append(0)
            elif picture[i][j] == 255:
                data.append([i, j])
                labels.append(1)

    # centre data
    data = np.array(data)
    mean = np.sum(data, 0) / data.shape[0]
    centred_data = (data - mean)

    return centred_data, np.array(labels)


def contour_polt(xx, yy, ax, clf):
    predictions = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    predictions = predictions.reshape(xx.shape)

    vmax = predictions.max()
    ax.contourf(xx, yy, predictions, vmin=-vmax, vmax=vmax, cmap="bwr", alpha=0.3, levels=100)


if __name__ == '__main__':
    data, labels = load_data_from_picture(cv.imread(DATA_PICTURE))

    # plot dataset
    plt.scatter(data[:, 0], data[:, 1], s=3, c=labels, cmap="bwr")
    plt.title("Used Dataset")
    plt.savefig("experiment_results/dataset.png")
    plt.close('all')

    # create grid to use in plots
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))

    # linear SVM
    # test different penalty parameters
    C = [0.001, 0.1, 10.0, 1000.0]
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(len(C)):
        ax = axs[i//2][i % 2]
        ax.set_title(f"C = {C[i]}")

        # fit classifier
        svm = SVC(kernel="linear", C=C[i])
        svm.fit(data, labels)

        # plot decision boundaries
        contour_polt(xx, yy, ax, svm)
        DecisionBoundaryDisplay.from_estimator(
            svm,
            data,
            plot_method="contour",
            colors="k",
            levels=[-1, 0, 1],
            alpha=0.5,
            linestyles=["--", "-", "--"],
            ax=ax,
        )
        # plot dataset
        ax.scatter(data[:, 0], data[:, 1], s=10, c=labels, cmap="bwr")
        # mark support vectors
        ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], s=100, marker='s', facecolors='none', edgecolors='black')

    fig.suptitle("Linear SVM")
    fig.tight_layout()
    plt.savefig("experiment_results/linearSVM.png")
    plt.close('all')

    # RBF SVM
    # test different Kernel coefficients
    gamma = ['scale', 0.00005, 0.0001, 0.001, 0.01, 0.1]
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    for i in range(len(gamma)):
        ax = axs[i//3][i % 3]
        ax.set_title(f"gamma = {gamma[i]}")
        if i==0:
            ax.set_title(f"gamma = {gamma[i]} ({1 / (2 * data.var())})")

        # fit classifier
        svm = SVC(kernel="rbf", gamma=gamma[i])
        svm.fit(data, labels)

        # plot decision boundaries
        contour_polt(xx, yy, ax, svm)
        DecisionBoundaryDisplay.from_estimator(
            svm,
            data,
            plot_method="contour",
            colors="k",
            levels=[-1, 0, 1],
            alpha=0.5,
            linestyles=["--", "-", "--"],
            ax=ax,
        )
        # plot dataset
        ax.scatter(data[:, 0], data[:, 1], s=10, c=labels, cmap="bwr")
        # mark support vectors
        ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], s=100, marker="s", facecolors='none', edgecolors='black')

    fig.suptitle("RBF kernel SVM")
    fig.tight_layout()
    plt.savefig("experiment_results/rbfSVM.png")
    plt.close('all')

    # Poly SVM
    # test different Kernel degrees and coefficients
    parameters = [(2, 0.1), (3, 0.1), (5, 0.1), (2, 5.0), (3, 5.0), (5, 5.0)]
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    for i in range(len(parameters)):
        ax = axs[i // 3][i % 3]
        ax.set_title(f"degree = {parameters[i][0]}, coefficient = {parameters[i][1]}")

        # fit classifier
        svm = SVC(kernel="poly", degree=parameters[i][0], coef0=parameters[i][1])
        svm.fit(data, labels)

        # plot decision boundaries
        contour_polt(xx, yy, ax, svm)
        DecisionBoundaryDisplay.from_estimator(
            svm,
            data,
            plot_method="contour",
            colors="k",
            levels=[-1, 0, 1],
            alpha=0.5,
            linestyles=["--", "-", "--"],
            ax=ax,
        )
        # plot dataset
        ax.scatter(data[:, 0], data[:, 1], s=10, c=labels, cmap="bwr")
        # mark support vectors
        ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], s=100, marker="s", facecolors='none', edgecolors='black')

    fig.suptitle("Polynomial kernel SVM")
    fig.tight_layout()
    plt.savefig("experiment_results/polySVM.png")
    plt.close('all')

    # custom kernel
    # K(x,y) = (R - ||x-y||)/R, if ||x-y|| < R , else 0
    R = 1
    def kernel(X, Y):
        distances = distance_matrix(X, Y)
        # multiply by matrix of booleans to filter values grater than R
        return (R - distances)/R*(distances <= R)

    # test different R params
    Rs = [3, 10, 20, 100]
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(len(Rs)):
        ax = axs[i//2][i % 2]
        ax.set_title(f"R = {Rs[i]}")

        # fit classifier
        R = Rs[i]
        svm = SVC(kernel=kernel)
        svm.fit(data, labels)

        # plot decision boundaries
        contour_polt(xx, yy, ax, svm)
        DecisionBoundaryDisplay.from_estimator(
            svm,
            data,
            plot_method="contour",
            colors="k",
            levels=[-1, 0, 1],
            alpha=0.5,
            linestyles=["--", "-", "--"],
            ax=ax,
        )
        # plot dataset
        ax.scatter(data[:, 0], data[:, 1], s=10, c=labels, cmap="bwr")

    fig.suptitle("Custom kernel SVM")
    fig.tight_layout()
    plt.savefig("experiment_results/customSVM.png")
    plt.close('all')


