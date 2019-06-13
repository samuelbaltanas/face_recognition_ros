""" Face extraction utilities using OpenPose and the COCO dataset.
    https://github.com/CMU-Perceptual-Computing-Lab/openpose
"""

import sys

import numpy as np
import cv2 as cv

from face_recognition_ros.extraction import bounding_box, face_extraction
from face_recognition_ros.utils import files

# Path in which openpose is installed after using `make install`
# You may comment this line if it is already included in PYTHONPATH
sys.path.append("/usr/local/python/")
from openpose import pyopenpose as op


class FacialDetector:
    def __init__(self, params):
        """ Initialize openpose

            params: Dict containing openpose config parameters
                https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/include/openpose/flags.hpp
        """
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

    def extract_faces(self, image):
        datum = self.extract_keypoints(image)

        bbs = bounding_box.extract_corners(datum)
        faces = face_extraction.extract_corners_image(image, bbs)

        return faces
