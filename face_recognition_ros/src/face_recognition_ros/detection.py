import numpy as np

from face_recognition_ros import datum
from face_recognition_ros.detectors import (
    mtcnn_mxnet_detector,
    opencv_face_detector,
)
from face_recognition_ros.utils import config

# mtcnn_face_detector,
# dlib_face_detector,
# openpose_face_detector,

METHODS = {
    # "dlib": dlib_face_detector.DlibDetector,
    # "mtcnn": mtcnn_face_detector.MtcnnFaceDetector,
    "mtcnn": mtcnn_mxnet_detector.MtcnnMxnetDetector,
    # "openpose": openpose_face_detector.FacialDetector,
    # "opencv": opencv_face_detector.OpencvFaceDetector,
}


class FaceDetector:
    def __init__(self, method=None, conf=None):

        # Parse config
        if conf is None:
            conf = config.CONFIG["DETECTION"]
        if method is None:
            method = conf["method"]
        if method in conf:
            method_conf = conf[method]
        else:
            method_conf = {}

        self.detector = METHODS[method](method_conf)

    def predict(
        self, X: np.ndarray, threshold=0.95, extract_image=False, **kwargs
    ) -> datum.Datum:
        return self.detector.predict(X, threshold, extract_image, **kwargs)
