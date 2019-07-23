import logging

import dlib

from face_recognition_ros.extraction import region
from face_recognition_ros.utils import files


DETECT_MODEL = files.get_model_path("mmod_human_face_detector.dat")
ALIGN_MODEL = files.get_model_path("shape_predictor_5_face_landmarks.dat")

logger = logging.getLogger(__name__)


class DlibDetector:
    def __init__(self):
        self.face_det = dlib.cnn_face_detection_model_v1(DETECT_MODEL)
        self.shape_pred = dlib.shape_predictor(ALIGN_MODEL)

    def extract_region(self, image):
        dets = self.face_det(image, 0)
        logger.info("Number of faces detected: {}".format(len(dets)))
        regions = [
            region.RectangleRegion(
                i.rect.left(),
                i.rect.top(),
                i.rect.right() - i.rect.left(),
                i.rect.bottom() - i.rect.top(),
            )
            for i in dets
        ]
        return regions

    def extract_images(self, image, align=True):
        dets = self.face_det(image, 0)
        logger.info("Number of faces detected: {}".format(len(dets)))
        if align:
            faces = dlib.full_object_detections()
            for detection in dets:
                faces.append(self.shape_pred(image, detection.rect))
            return dlib.get_face_chips(image, faces, size=320)
        else:
            res = []
            for det in dets:
                rec = det.rect
                res.append(image[rec.top() : rec.bottom(), rec.left() : rec.right()])
            return res
