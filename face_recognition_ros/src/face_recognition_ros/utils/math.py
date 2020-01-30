import numpy as np
from sklearn import preprocessing


def dist(x1, x2, func=0):
    if func == 0:
        y = np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
    elif func == 1:
        dot = np.sum(np.multiply(x1, x2), axis=1)
        norm = np.linalg.norm(x1, axis=1) * np.linalg.norm(x2, axis=1)
        similarity = dot / norm
        y = np.arccos(similarity) / np.pi
    else:
        raise RuntimeError("Distance function not defined")
    return y


def rotate(point, angle, origin):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return (R.dot(point - origin)) + origin


def solve_line_intersect(p1, p2, a, b):
    p1 = np.vstack(p1)
    p2 = np.vstack(p2)
    A = np.vstack([a, -1 * b])
    A = A.T
    B = p2 - p1
    C = np.linalg.solve(A, B)
    d = C[0, 0] * np.vstack(a) + p1

    return d.flatten()


def compute_centroid(embeddings, labels, func=1):
    emb = []
    lab = np.unique(labels)

    for i in lab:
        is_im = i == labels
        x1 = embeddings[is_im]
        x2 = x1.mean(axis=0).reshape(1, -1)
        if func == 1:
            x2 = preprocessing.normalize(x2)

        emb.append(x2)

        return np.vstack(emb), lab
