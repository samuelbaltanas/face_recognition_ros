from __future__ import print_function

import numpy as np
from face_recognition_ros.utils.math import dist


def comparison_matrix(embedings, func=0):
    n_imgs = embedings.shape[0]
    mat = np.empty((n_imgs, n_imgs))
    for i, emb1 in enumerate(embedings):
        for j, emb2 in enumerate(embedings):
            mat[i, j] = dist(emb1, emb2, func)
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
