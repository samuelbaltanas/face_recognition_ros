from face_recognition_ros.utils import config
from face_recognition_ros.detectors import dlib_face_detector, mtcnn_face_detector, opencv_face_detector, openpose_face_detector


_METHODS = {
    "dlib": dlib_face_detector.DlibDetector,
    "mtcnn": mtcnn_face_detector.MtcnnFaceDetector,
    "openpose": openpose_face_detector.FacialDetector,
    "opencv": opencv_face_detector.OpencvFaceDetector
}


class FacialDetector:
    def __init__(self, method="mtcnn", conf=None):
        if conf is None:
            conf = config.CONFIG

        self.detector = _METHODS[method](conf)

    def extract_region(self, image):
        return self.detector.extract_region(image)

    def extract_images(self, image):
        return self.detector.extract_images(image)
