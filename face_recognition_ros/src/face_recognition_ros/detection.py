from face_recognition_ros.core import datum
from face_recognition_ros.utils import config
from face_recognition_ros.detectors import mtcnn_mxnet_detector, opencv_face_detector

# mtcnn_face_detector,
# dlib_face_detector,
# openpose_face_detector,

METHODS = {
    # "dlib": dlib_face_detector.DlibDetector,
    # "mtcnn": mtcnn_face_detector.MtcnnFaceDetector,
    "mtcnn": mtcnn_mxnet_detector.MtcnnMxnetDetector,
    # "openpose": openpose_face_detector.FacialDetector,
    "opencv": opencv_face_detector.OpencvFaceDetector,
}


class FacialDetector:
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

    def extract_region(self, image, threshold=0.9):
        return self.detector.extract_region(image, threshold)

    def extract_images(self, image, regions=None, raw_detection=None, threshold=0.9):
        return self.detector.extract_images(image, regions, raw_detection, threshold)

    def extract_datum(self, image, threshold=0.9):
        regions, raw_detection = self.detector.extract_region(image, threshold)
        images = self.detector.extract_images(image, regions, raw_detection)

        data = [
            datum.Datum(face_region=reg, face_image=im)
            for reg, im in zip(regions, images)
        ]

        return data
