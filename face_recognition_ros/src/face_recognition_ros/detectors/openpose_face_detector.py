""" Face extraction utilities using OpenPose and the COCO dataset.
    https://github.com/CMU-Perceptual-Computing-Lab/openpose
"""

import sys

import numpy as np
from scipy.spatial.distance import euclidean as dist

from face_recognition_ros.core import region
from face_recognition_ros.detectors import base_face_detector

# Path in which openpose is installed after using `make install`
# You may comment this line if it is already included in PYTHONPATH

# sys.path.append('/usr/local/python')

from openpose import pyopenpose as op  # noqa: E402

threshold = 0.15


# DONE: Change interface
class FacialDetector(base_face_detector.BaseFaceDetector):
    def __init__(self, conf=None):

        self.opwrapper = op.WrapperPython()
        self.opwrapper.configure(conf)
        self.opwrapper.start()

    def extract_region(self, image):
        datum = op.Datum()
        datum.cvInputData = image
        try:
            self.opwrapper.emplaceAndPop([datum])
        except Exception as e:
            print(e)
            sys.exit(-1)
        else:
            regions = [
                openpose_face_detector2(pose, threshold) for pose in datum.poseKeypoints
            ]

            regions = [reg for reg in regions if not np.any(reg.dimensions == 0)]

        return regions, datum


def openpose_face_detector(posePtr, threshold):
    point_top_left = np.zeros(2)
    face_size = 0.0
    score = 0.0

    points_used = set()

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
                points_used = points_used.union([0, 14, 16])
            else:
                point_top_left += (
                    posePtr[15, 0:2] + posePtr[17, 0:2] + posePtr[0, 0:2]
                ) / 3.0
                face_size += 0.85 * (
                    dist(posePtr[15, 0:2], posePtr[17, 0:2])
                    + dist(posePtr[0, 0:2], posePtr[17, 0:2])
                    + dist(posePtr[15, 0:2], posePtr[0, 0:2])
                )
                points_used = points_used.union([0, 15, 17])

        else:
            point_top_left += (posePtr[1, 0:2] + posePtr[0, 0:2]) / 2.0
            face_size += 2.0 * dist(posePtr[1, 0:2], posePtr[0, 0:2])
            points_used = points_used.union([0, 1])

        counter += 1.0

    if lEyeScoreAbove and rEyeScoreAbove:
        point_top_left += (posePtr[14, 0:2] + posePtr[15, 0:2]) / 2.0
        face_size += 3.0 * dist(posePtr[14, 0:2], posePtr[15, 0:2])
        counter += 1.0
        points_used = points_used.union([14, 15])

    if lEarScoreAbove and rEarScoreAbove:
        point_top_left += (posePtr[16, 0:2] + posePtr[17, 0:2]) / 2.0
        face_size += 2.0 * dist(posePtr[16, 0:2], posePtr[17, 0:2])
        counter += 1.0
        points_used = points_used.union([16, 17])

    if counter > 0:
        point_top_left /= counter
        face_size /= counter
        score = np.mean(posePtr[list(points_used), 2])

    return region.RectangleRegion(
        point_top_left[0] - face_size / 2,
        point_top_left[1] - face_size / 2,
        face_size,
        face_size,
        score,
    )


def openpose_face_detector2(posePtr, threshold):
    point_top_left = np.zeros(2)
    face_size = [0.0, 0.0]  # (width, height)
    score = np.mean(posePtr[[0, 1, 14, 15, 16, 17], 2])

    l_true = None
    r_true = None

    neckScoreAbove = posePtr[1, 2] > threshold
    headNoseScoreAbove = posePtr[0, 2] > threshold
    lEarScoreAbove = posePtr[16, 2] > threshold
    rEarScoreAbove = posePtr[17, 2] > threshold
    lEyeScoreAbove = posePtr[14, 2] > threshold
    rEyeScoreAbove = posePtr[15, 2] > threshold

    counter = 0.2

    # Leftmost point
    if lEarScoreAbove:
        l_true = posePtr[16, 0:2]
    # Rightmost point
    if rEarScoreAbove:
        r_true = posePtr[17, 0:2]

    counter = 0.0

    if headNoseScoreAbove:
        if (
            lEyeScoreAbove == lEarScoreAbove
            and rEyeScoreAbove == rEarScoreAbove
            and lEyeScoreAbove != rEyeScoreAbove
        ):
            if lEyeScoreAbove:
                point_top_left += (
                    posePtr[14, 0:2] + posePtr[16, 0:2] + posePtr[0, 0:2]
                ) / 3.0
                face_size[1] += 0.85 * (
                    dist(posePtr[14, 0:2], posePtr[16, 0:2])
                    + dist(posePtr[0, 0:2], posePtr[16, 0:2])
                    + dist(posePtr[14, 0:2], posePtr[0, 0:2])
                )
                face_size[0] += 0.85 * (
                    dist(posePtr[14, 0:2], posePtr[16, 0:2])
                    + dist(posePtr[0, 0:2], posePtr[14, 0:2])
                )
            else:
                point_top_left += (
                    posePtr[15, 0:2] + posePtr[17, 0:2] + posePtr[0, 0:2]
                ) / 3.0
                face_size[1] += 0.85 * (
                    dist(posePtr[15, 0:2], posePtr[17, 0:2])
                    + dist(posePtr[0, 0:2], posePtr[17, 0:2])
                    + dist(posePtr[15, 0:2], posePtr[0, 0:2])
                )
                face_size[0] += 0.85 * (
                    dist(posePtr[15, 0:2], posePtr[17, 0:2])
                    + dist(posePtr[0, 0:2], posePtr[15, 0:2])
                )
            counter += 1.0
        elif neckScoreAbove:
            point_top_left += posePtr[0, 0:2]
            face_size[1] += 1.0 * dist(posePtr[1, 0:2], posePtr[0, 0:2])
            face_size[0] += 0.0 * dist(posePtr[1, 0:2], posePtr[0, 0:2])
            counter += 1.0

    if lEyeScoreAbove and rEyeScoreAbove:
        point_top_left += (posePtr[14, 0:2] + posePtr[15, 0:2]) / 2.0
        face_size[1] += 4.0 * dist(posePtr[14, 0:2], posePtr[15, 0:2])
        face_size[0] += 3.0 * dist(posePtr[14, 0:2], posePtr[15, 0:2])
        counter += 1.0

    if lEarScoreAbove and rEarScoreAbove:
        point_top_left += (posePtr[16, 0:2] + posePtr[17, 0:2]) / 2.0
        face_size[1] += 1.2 * dist(posePtr[16, 0:2], posePtr[17, 0:2])
        face_size[0] += dist(posePtr[16, 0:2], posePtr[17, 0:2])
        counter += 1.0

    if counter > 0:
        point_top_left /= counter
        face_size[0] /= counter
        face_size[1] /= counter

    if l_true is not None:
        ll = l_true[0]
        face_size[0] = face_size[0] - (ll - (point_top_left[0] - face_size[0] / 2))
    else:
        ll = point_top_left[0] - face_size[0] / 2
    #
    if r_true is not None:
        rr = r_true[0]
        face_size[0] = rr - ll

    return region.RectangleRegion(
        ll, point_top_left[1] - face_size[1] / 2, face_size[0], face_size[1], score
    )
