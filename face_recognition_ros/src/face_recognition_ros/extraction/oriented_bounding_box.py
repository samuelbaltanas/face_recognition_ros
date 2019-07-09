""" Face extraction arbitrarily oriented bounding boxes.

    It uses the position of the eyes or the ears for alignment.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt


# DONE: Separate different types
# DONE: Extract info of which pose correspond to which box
# TODO: Keypoints are reversed sometimes (RL or LR)
# TODO: Missing side face extraction (1 eye missing)
# TODO: Incorporate ideas from https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/face/faceDetector.cpp
# DONE: Frontal face (Both sides available)
# DONE: Partial side face (Both eyes available)


def extract_from_pose(datum, confidence_threshold=0.00):
    bbs = []
    for idx, p in enumerate(datum.poseKeypoints):
        bb = extraction_method_1(p, confidence_threshold)
        if bb is not None:
            bbs.append((idx, bb))

    return bbs


def plot_bounding_boxes(image, bbs, color_enc=None, **kwargs):
    plt.figure()

    if color_enc is not None:
        image = cv2.cvtColor(image, color_enc)

    for _, bb in bbs:
        bb = bb.T
        plt.scatter(bb[0], bb[1], **kwargs)

    plt.imshow(image)


def extraction_method_1(p, confidence_threshold):
    pres = p[:, 2] > confidence_threshold

    bb = None

    if not np.any(pres[[14, 15, 0]]):
        pass
    elif pres[17] and pres[16]:
        # Best condition: two sides present
        ratio = 20.9 / 16.1

        # Side to side vector
        v = (p[17, 0:2] - p[16, 0:2]) / 2
        vt = ratio * v[::-1]
        vt[0] *= -1

        # Corners
        # center = np.mean(p[[0, 16, 17, 14, 15], 0:2], axis=0)
        topl = p[16, 0:2] - vt
        botr = p[17, 0:2] + vt
        topr = p[17, 0:2] - vt
        botl = p[16, 0:2] + vt

        bb = np.float32([topl, topr, botl, botr])
    elif pres[14] and pres[15] and pres[0] and pres[1]:
        if pres[16]:
            orig = p[16, 0:2]
            d = -1
        elif pres[17]:
            orig = p[17, 0:2]
            d = 1
        else:
            return

        rem_idx = np.array([0, 14, 15])
        rem = p[rem_idx, 0:2]

        furth = np.argmax(np.sum((orig - rem) ** 2, axis=1))

        # print(np.sum((p[[pts[0]], 0:2] - p[pts[1:], 0:2])**2, axis=1))
        v = rem[1] - rem[2]
        vs = np.linalg.norm(v)
        v /= vs

        vt = np.copy(v[::-1])
        vt[1] = vt[1] * -1

        upvec = (p[0, 0:2] - p[1, 0:2]) / 2

        a = orig
        b = -1.2 * upvec + rem[0]
        c = rem[furth] + d * v * vs / 2
        d = 1.2 * upvec + rem[0]

        botl = solve_line_intersect(a, b, vt, v)
        botr = solve_line_intersect(b, c, v, vt)
        topr = solve_line_intersect(c, d, vt, v)
        topl = solve_line_intersect(d, a, v, vt)

        bb = np.float32([topl, topr, botl, botr])
    else:
        pass

    return bb


def solve_line_intersect(p1, p2, a, b):
    p1 = np.vstack(p1)
    p2 = np.vstack(p2)
    A = np.vstack([a, -1 * b])
    A = A.T
    B = p2 - p1
    C = np.linalg.solve(A, B)
    d = C[0, 0] * np.vstack(a) + p1

    return d.flatten()
