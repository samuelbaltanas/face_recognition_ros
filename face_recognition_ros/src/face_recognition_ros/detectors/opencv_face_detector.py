import logging

import cv2

from face_recognition_ros import region
from face_recognition_ros.detectors import base_face_detector
from face_recognition_ros.utils import files


class OpencvFaceDetector(base_face_detector.BaseFaceDetector):
    def __init__(self, conf):
        self.threshold = conf["threshold"]
        self.margin = conf["margin"]

        MODEL_FILE = files.get_model_path(conf["path"], conf["model"])
        CONFIG_FILE = files.get_model_path(conf["path"], conf["config"])
        self.net = cv2.dnn.readNetFromTensorflow(MODEL_FILE, CONFIG_FILE)

    def extract_region(self, image):
        DIMS = image.shape
        blob = cv2.dnn.blobFromImage(
            image, 1.0, (300, 300)  # , [104, 117, 123], False, False
        )
        self.net.setInput(blob)
        detections = self.net.forward()[0, 0, :, :]
        regions = []

        for i in range(detections.shape[0]):
            if detections[i, 2] >= self.threshold:
                reg = region.BoundingBox(
                    detections[i, 3] * DIMS[1] - self.margin / 2,
                    detections[i, 4] * DIMS[0] - self.margin / 2,
                    (detections[i, 5] - detections[i, 3]) * DIMS[1] + self.margin,
                    (detections[i, 6] - detections[i, 4]) * DIMS[0] + self.margin,
                    detections[i, 2],
                )
                logging.debug("Face {} detected: {}".format(i, str(reg)))
                regions.append(reg)

        return regions, detections
