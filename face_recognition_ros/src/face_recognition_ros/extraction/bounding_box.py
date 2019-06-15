import numpy as np
from scipy.spatial.distance import cdist


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


def extract_corners(datum, confidence_threshold=0.0):
    bbs = []
    for person in datum.poseKeypoints:
        if np.all(person[[16, 17, 0], 2] > confidence_threshold):
            # Best condition: two sides present
            ratio = 20.9 / 16.1
            # Side to side vector
            v = person[17, 0:2] - person[16, 0:2]
            vt = -ratio * v[::-1]

            # Corners
            center = np.mean(person[[0, 16, 17, 14, 15], 0:2], axis=0)
            topl = center + (vt - v) / 2
            botr = center + (v - vt) / 2
            topr = center + (v + vt) / 2
            botl = center - (v + vt) / 2
            bb = np.float32([topl, topr, botl, botr])

            bbs.append(bb)
        elif np.all(person[[15, 17, 0], 2] > confidence_threshold) or np.all(
            person[[14, 16, 0], 2] > confidence_threshold
        ):
            # Worse condition: one side only
            leftie = np.all(person[[15, 17, 0], 2] > confidence_threshold)
            pts = [16, 0, 14, 15] if leftie else [17, 0, 15, 14]

            center = np.mean(person[pts, 0:2], axis=0)
            leftmost_idx = np.argmax(
                cdist(person[pts[0], 0:2], person[pts[1:], 0:2])
            )

            v = person[pts[0], 0:2] - person[leftmost_idx, 0:2]

    return bbs
