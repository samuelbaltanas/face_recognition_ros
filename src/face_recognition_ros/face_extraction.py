''' Face extraction utilities using OpenPose and the COCO dataset.
    https://github.com/CMU-Perceptual-Computing-Lab/openpose

    - TODO Profile performance and speed up procedure
        https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/speed_up_openpose.md
    - TODO Configuration of openpose
'''

import sys
from utils import files

# Path in which openpose is installed after using `make install`
# You may comment this line if it is already included in PYTHONPATH
sys.path.append('/usr/local/python/')
from openpose import pyopenpose as op


class FacialDetector():
    def __init__(self, params):
        """ Initialize openpose

            params: Dict containing openpose config parameters
                https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/include/openpose/flags.hpp
        """
        self.opwrapper = op.WrapperPython()
        self.opwrapper.configure(params)

    def extract_keypoints(self, image):
        """ Feed image to openpose and return face-related keypoints
            https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md

            Useful keypoints in COCO dataset:
                - 1: neck
                - 0: nose
                - 14/15: left/right eyes
                - 16/17: left/right sides of head

            Params:
                image: opencv image
       """
        datum = op.Datum()
        datum.cvInputData = image
        try:
            self.opwrapper.waitAndPop([datum])
        except Exception as e:
            print(e)
            sys.exit(-1)

        keypoints = (datum.poseIds, datum.poseKeypoints)

        # TODO Select keypoints
        pass

        return keypoints

    def __enter__(self):
        self.opwrapper.start()

    def __exit__(self, type, value, traceback):
        self.opwrapper.stop()


def extract_bounding_box(image, keypoints):
    """ Extract face from image and return bounding box
    """
    # TODO Extract bounding box
    pass
