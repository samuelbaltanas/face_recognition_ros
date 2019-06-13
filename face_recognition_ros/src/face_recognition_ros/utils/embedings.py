from __future__ import print_function

import numpy as np


def compare_images(embedings):
    n_imgs = embedings.shape[0]

    mat = np.empty((n_imgs, n_imgs))
    for i in range(n_imgs):
        for j in range(n_imgs):
            mat[i, j] = np.sqrt(
                np.sum((embedings[i, :] - embedings[j, :]) ** 2)
            )
    return mat


def plot_comp_matrix(mat):
    n_imgs = mat.shape[0]
    print("    ", end="")
    for i in range(n_imgs):
        print("    %1d     " % i, end="")
    print("")
    for i in range(n_imgs):
        print("%1d  " % i, end="")
        for j in range(n_imgs):
            print("  %1.4f  " % mat[i, j], end="")
        print("")
