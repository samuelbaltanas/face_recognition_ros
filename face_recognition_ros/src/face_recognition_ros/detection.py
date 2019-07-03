""" Face extraction utilities using OpenPose and the COCO dataset.
    https://github.com/CMU-Perceptual-Computing-Lab/openpose
"""

import sys

import numpy as np
import cv2 as cv

from face_recognition_ros.extraction import (
    oriented_bounding_box,
    aligned_bounding_box,
)
from face_recognition_ros.utils import files, config

# Path in which openpose is installed after using `make install`
# You may comment this line if it is already included in PYTHONPATH
sys.path.append("/usr/local/python/")
from openpose import pyopenpose as op


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

    def extract_keypoints(self, image):
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

        return datum

    def extract_faces(self, image, method=0):
        datum = self.extract_keypoints(image)

        # DONE: Method selection
        # TODO: output additional information
        if method == 0:
            bbs = oriented_bounding_box.extract_from_pose(datum)
            faces = oriented_bounding_box.extract_image_faces(bbs, image)
        elif method == 1:
            bbs = aligned_bounding_box.extract_from_pose(datum)
            faces = aligned_bounding_box.extract_image_faces(bbs, image)

        return faces
