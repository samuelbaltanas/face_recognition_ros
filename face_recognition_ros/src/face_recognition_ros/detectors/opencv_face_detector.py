import logging

import cv2

from face_recognition_ros.extraction import region
from face_recognition_ros.utils import files

MODEL_FILE = files.get_model_path("opencv_face_detector_uint8.pb")
CONFIG_FILE = files.get_model_path("opencv_face_detector.pbtxt")


class OpencvFaceDetector:
    def __init__(self):
        self.net = cv2.dnn.readNetFromTensorflow(MODEL_FILE, CONFIG_FILE)

    def extract_region(self, image, threshold=0.15):
        DIMS = image.shape
        blob = cv2.dnn.blobFromImage(
            image, 1.0, (300, 300)  # [104, 117, 123], False, False
        )
        self.net.setInput(blob)
        detections = self.net.forward()
        regions = []

        for i in range(detections.shape[2]):
            if detections[0, 0, i, 2] >= threshold:
                reg = region.RectangleRegion(
                    detections[0, 0, i, 3] * DIMS[1],
                    detections[0, 0, i, 4] * DIMS[0],
                    (detections[0, 0, i, 5] - detections[0, 0, i, 3]) * DIMS[1],
                    (detections[0, 0, i, 6] - detections[0, 0, i, 4]) * DIMS[0],
                    detections[0, 0, i, 2],
                )
                logging.debug("Face {} detected: {}".format(i, str(reg)))
                regions.append(reg)

        return regions

    def extract_images(self, image):
        regions = self.extract_region(image)
        return [r.extract_face(image) for r in regions]
