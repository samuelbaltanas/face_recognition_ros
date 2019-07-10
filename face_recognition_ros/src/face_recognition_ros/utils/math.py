import numpy as np


def dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


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
