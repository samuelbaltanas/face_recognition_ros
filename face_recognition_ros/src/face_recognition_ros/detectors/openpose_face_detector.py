""" Face extraction utilities using OpenPose and the COCO dataset.
    https://github.com/CMU-Perceptual-Computing-Lab/openpose
"""

import sys
import logging

import numpy as np

# from face_recognition_ros.extraction import oriented_bounding_box, aligned_bounding_box
from face_recognition_ros.utils import config
from face_recognition_ros.utils.math import dist
from face_recognition_ros.extraction import region
from face_recognition_ros.datum import Datum
# from face_recognition_ros.extraction import methods

# Path in which openpose is installed after using `make install`
# You may comment this line if it is already included in PYTHONPATH
sys.path.append("/usr/local/python/")
from openpose import pyopenpose as op  # noqa: E402

logger = logging.getLogger(__name__)


# TODO: Change interface
class FacialDetector:
    def __init__(self, conf=None):
        """ Initialize openpose

            params: Dict containing openpose config parameters
                https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/include/openpose/flags.hpp
        """

        if conf is None:
            conf = config.CONFIG

        params = conf["OPENPOSE"]

        self.opwrapper = op.WrapperPython()
        self.opwrapper.configure(params)
        self.opwrapper.start()

    def extract(self, image, threshold):
        """ Feed image to openpose and return face-related keypoints
            https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md


            Params:
                image: opencv image
       """
        datum = op.Datum()
        datum.cvInputData = image
        try:
            self.opwrapper.emplaceAndPop([datum])
        except Exception as e:
            print(e)
            sys.exit(-1)

        res = []
        for idx, (pose, score) in enumerate(zip(datum.poseKeypoints, datum.poseScores)):
            dat = Datum(
                pose=pose,
                pose_score=score,
                face_region=openpose_face_detector(pose, threshold),
            )

            if not dat.face_region.is_empty():
                logger.debug(
                    "Face detected in pose {}. Region (rectangle): {}".format(
                        idx, repr(dat.face_region)
                    )
                )
                try:
                    dat.face_image = dat.face_region.extract_face(image)
                except Exception:
                    logger.warn(
                        "Exception in pose {}. Region (rectangle): {}".format(
                            idx, repr(dat.face_region)
                        )
                    )
                    continue
                res.append(dat)
            else:
                logger.debug(
                    "Face skipped in pose {}. Region (rectangle): {}".format(
                        idx, repr(dat.face_region)
                    )
                )

        logging.info("Faces detected: {}".format(len(res)))
        return res


def openpose_face_detector(posePtr, threshold):
    # TODO: Divide face size in v_size and h_size
    point_top_left = np.zeros(2)
    face_size = 0.0
    score = np.mean(posePtr[[0, 1, 14, 15, 16, 17], 2])

    neckScoreAbove = posePtr[1, 2] > threshold
    headNoseScoreAbove = posePtr[0, 2] > threshold
    lEarScoreAbove = posePtr[16, 2] > threshold
    rEarScoreAbove = posePtr[17, 2] > threshold
    lEyeScoreAbove = posePtr[14, 2] > threshold
    rEyeScoreAbove = posePtr[15, 2] > threshold

    counter = 0.0

    if lEyeScoreAbove and lEarScoreAbove and headNoseScoreAbove and not rEarScoreAbove:
        point_top_left += (posePtr[14, 0:2] + posePtr[16, 0:2] + posePtr[0, 0:2]) / 3.0
        face_size += 0.85 * (
            dist(posePtr[14, 0:2], posePtr[16, 0:2])
            + dist(posePtr[0, 0:2], posePtr[16, 0:2])
            + dist(posePtr[14, 0:2], posePtr[0, 0:2])
        )
        counter += 1.0

    if rEyeScoreAbove and rEarScoreAbove and headNoseScoreAbove and not lEarScoreAbove:
        point_top_left += (posePtr[15, 0:2] + posePtr[17, 0:2] + posePtr[0, 0:2]) / 3.0
        face_size += 0.85 * (
            dist(posePtr[15, 0:2], posePtr[17, 0:2])
            + dist(posePtr[0, 0:2], posePtr[17, 0:2])
            + dist(posePtr[15, 0:2], posePtr[0, 0:2])
        )
        counter += 1.0

    if neckScoreAbove and headNoseScoreAbove:
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

    return region.RectangleRegion(
        point_top_left[0] - face_size / 2,
        point_top_left[1] - face_size / 2,
        face_size,
        face_size,
        score,
    )
