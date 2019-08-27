import logging

import dlib

from face_recognition_ros.detectors import base_face_detector
from face_recognition_ros.core import region
from face_recognition_ros.utils import files


DETECT_MODEL = files.get_model_path("", "mmod_human_face_detector.dat")
ALIGN_MODEL = files.get_model_path("", "shape_predictor_5_face_landmarks.dat")

logger = logging.getLogger(__name__)


class DlibDetector(base_face_detector.BaseFaceDetector):
    def __init__(self, conf):
        # TODO: Use config ALIGN
        self.face_det = dlib.cnn_face_detection_model_v1(DETECT_MODEL)
        self.shape_pred = dlib.shape_predictor(ALIGN_MODEL)

    def extract_region(self, image):
        raw_detection = self.face_det(image, 0)
        logger.info("Number of faces detected: {}".format(len(raw_detection)))
        regions = [
            region.RectangleRegion(
                i.rect.left(),
                i.rect.top(),
                i.rect.right() - i.rect.left(),
                i.rect.bottom() - i.rect.top(),
            )
            for i in raw_detection
        ]
        return regions, raw_detection

    def extract_images(self, image, regions=None, raw_detection=None, align=True):
        if raw_detection is None:
            regions, raw_detection = self.face_det(image, 0)

        if align:
            if len(regions) > 0:
                faces = dlib.full_object_detections()
                for detection in raw_detection:
                    faces.append(self.shape_pred(image, detection.rect))
                return dlib.get_face_chips(
                    image, faces, size=160
                )  # TODO: incorporate size and padding
            else:
                return []
        else:
            return super().extract_images(image, regions, raw_detection)
