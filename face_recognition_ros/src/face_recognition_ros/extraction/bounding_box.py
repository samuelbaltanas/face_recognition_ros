""" Bounding box extraction


"""
import numpy as np

# from scipy.spatial.distance import cdist

# TODO: Change to classes
# TODO: Extract info of which pose correspond to which box
# TODO: Keypoints are reversed sometimes (RL or LR)
# TODO: Missing side face extraction (1 eye missing)
# TODO: Incorporate ideas from https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/face/faceDetector.cpp
# DONE: Frontal face (Both sides available)
# DONE: Partial side face (Both eyes available)


def extract_bounding_boxes(datum, confidence_threshold=0.01):
    """ Extract faces from image and return bounding boxes

        Useful keypoints in COCO dataset:
            - 1: neck
            - 0: nose
            - 14/15: left/right eyes
            - 16/17: left/right sides of head
    """
    REQ_KEYPOINTS = [16, 17]

    bbs = []
    for person in datum.poseKeypoints:
        bb = np.zeros(4, dtype=int)

        # Skip incomplete poses (we cannot extract a fake)
        if np.any(person[REQ_KEYPOINTS, 2] <= confidence_threshold):
            # print person[[16, 17], 2]
            continue

        feats = person[[0, 1, 16, 17], 0:2]
        vec = (20.9 / 16.1) * (feats[3] - feats[2]) / 2

        top = feats[0] - vec[::-1]
        bot = feats[0] + vec[::-1]

        bb[0] = int(top[1])
        bb[1] = int(bot[1])
        bb[2] = int(feats[2, 0])
        bb[3] = int(feats[3, 0])

        bbs.append(bb)

    return bbs


def extract_corners(datum, confidence_threshold=0.00):
    bbs = []
    for p in datum.poseKeypoints:
        pres = p[:, 2] > confidence_threshold

        if not np.any(pres[[14, 15, 0]]):
            continue
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
        else:
            continue

        bb = np.float32([topl, topr, botl, botr])
        bbs.append(bb)

    return bbs


def solve_line_intersect(p1, p2, a, b):
    p1 = np.vstack(p1)
    p2 = np.vstack(p2)
    A = np.vstack([a, -1 * b])
    A = A.T
    B = p2 - p1
    C = np.linalg.solve(A, B)
    d = C[0, 0] * np.vstack(a) + p1

    return d.flatten()
