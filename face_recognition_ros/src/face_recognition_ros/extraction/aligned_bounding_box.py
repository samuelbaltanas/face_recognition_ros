""" Functions to extract faces to axis-aligned bounding boxes """
import numpy as np
import cv2
import matplotlib.pyplot as plt

# TODO: Enumerate


def extract_from_pose(datum, confidence_threshold=0.01):

    bbs = []

    imgshape = datum.cvOutputData.shape

    for p in datum.poseKeypoints:
        # bb = sample_face_detector(p, confidence_threshold)
        bb = openpose_face_detector(p, confidence_threshold)

        if np.all(bb == 0):
            continue

        assert bb[0] < bb[1] and bb[2] < bb[3], "No size"

        bb2 = np.zeros(4, dtype=int)
        bb2[0] = int(min(imgshape[0] - 1, max(0, bb[0])))
        bb2[1] = int(min(imgshape[0] - 1, max(0, bb[1])))
        bb2[2] = int(min(imgshape[1] - 1, max(0, bb[2])))
        bb2[3] = int(min(imgshape[1] - 1, max(0, bb[3])))

        if bb2[0] >= bb2[1] or bb2[2] >= bb2[3]:
            continue

        bbs.append(bb2)

    return bbs


def sample_face_detector(p, confidence_threshold):
    REQ_KEYPOINTS = [16, 17]
    bb = np.zeros(4, dtype=int)

    # Skip incomplete poses (we cannot extract a fake)
    if np.any(p[REQ_KEYPOINTS, 2] <= confidence_threshold):
        # print person[[16, 17], 2]
        return None

    feats = p[[0, 1, 16, 17], 0:2]
    vec = (20.9 / 16.1) * (feats[3] - feats[2]) / 2

    top = feats[0] - vec[::-1]
    bot = feats[0] + vec[::-1]

    bb[0] = int(top[1])
    bb[1] = int(bot[1])
    bb[2] = int(feats[2, 0])
    bb[3] = int(feats[3, 0])

    return bb


def dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def openpose_face_detector(posePtr, threshold):
    point_top_left = np.zeros(2)
    face_size = 0.0

    neckScoreAbove = posePtr[1, 2] > threshold
    headNoseScoreAbove = posePtr[0, 2] > threshold
    lEarScoreAbove = posePtr[16, 2] > threshold
    rEarScoreAbove = posePtr[17, 2] > threshold
    lEyeScoreAbove = posePtr[14, 2] > threshold
    rEyeScoreAbove = posePtr[15, 2] > threshold

    counter = 0.0

    if neckScoreAbove and headNoseScoreAbove:
        if (
            lEyeScoreAbove == lEarScoreAbove
            and rEyeScoreAbove == rEarScoreAbove
            and lEyeScoreAbove != rEyeScoreAbove
        ):
            if lEyeScoreAbove:
                point_top_left += (
                    posePtr[14, 0:2] + posePtr[16, 0:2] + posePtr[0, 0:2]
                ) / 3.0
                face_size += 0.85 * (
                    dist(posePtr[14, 0:2], posePtr[16, 0:2])
                    + dist(posePtr[0, 0:2], posePtr[16, 0:2])
                    + dist(posePtr[14, 0:2], posePtr[0, 0:2])
                )
            else:
                point_top_left += (
                    posePtr[15, 0:2] + posePtr[17, 0:2] + posePtr[0, 0:2]
                ) / 3.0
                face_size += 0.85 * (
                    dist(posePtr[15, 0:2], posePtr[17, 0:2])
                    + dist(posePtr[0, 0:2], posePtr[17, 0:2])
                    + dist(posePtr[15, 0:2], posePtr[0, 0:2])
                )
        else:
            point_top_left += (posePtr[1, 0:2] + posePtr[0, 0:2]) / 2.0
            face_size += 2.0 * dist(posePtr[1, 0:2], posePtr[0, 0:2])

        counter += 1.0

    if lEyeScoreAbove and rEyeScoreAbove:
        point_top_left += (posePtr[14, 0:2] + posePtr[15, 0:2]) / 2.0
        face_size += 3.0 * dist(posePtr[14, 0:2], posePtr[15, 0:2])
        counter += 1.0

    if lEarScoreAbove and rEarScoreAbove:
        point_top_left += (posePtr[16, 0:2] + posePtr[17, 0:2]) / 2.0
        face_size += 2.0 * dist(posePtr[16, 0:2], posePtr[17, 0:2])
        counter += 1.0

    if counter > 0:
        point_top_left /= counter
        face_size /= counter

    return np.array(
        [
            point_top_left[1] - face_size / 2,
            point_top_left[1] + face_size / 2,
            point_top_left[0] - face_size / 2,
            point_top_left[0] + face_size / 2,
        ]
    )


def extract_image_faces(bbs, image):
    faces = []

    for bb in bbs:
        face = image[bb[0] : bb[1], bb[2] : bb[3]]
        face = cv2.resize(face, (160, 160), interpolation=cv2.INTER_AREA)
        faces.append(face)

    return faces


def plot_bounding_box(
    image, bbs, color=(0, 255, 0), color_enc=None, thickness=10
):
    imbb = image.copy()

    if color_enc is not None:
        imbb = cv2.cvtColor(imbb, color_enc)

    for bb in bbs:
        imbb = cv2.rectangle(
            imbb, (bb[2], bb[0]), (bb[3], bb[1]), color, thickness=thickness
        )

    plt.imshow(imbb)
    return imbb
