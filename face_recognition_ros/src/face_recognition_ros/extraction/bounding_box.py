import numpy as np


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
        vec = (20.9/16.1)*(feats[3] - feats[2])/2

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
        REQ_KEYPOINTS = [16, 17]

        if np.any(person[REQ_KEYPOINTS, 2] <= confidence_threshold):
            # print person[[16, 17], 2]
            continue

        fts = person[[0, 1, 16, 17], 0:2]
        vec = (20.9/16.1)*(fts[3] - fts[2])/2

        rev = vec[::-1]
        rev[0] *= -1

        topl = fts[2] - rev
        botr = fts[3] + rev

        topr = fts[3] - rev
        botl = fts[2] + rev

        bb = np.float32([topl, topr, botl, botr])

        bbs.append(bb)

    return bbs
