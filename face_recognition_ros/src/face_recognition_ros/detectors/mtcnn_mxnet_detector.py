# coding: utf-8
from face_recognition_ros.core import region
from face_recognition_ros.detectors import base_face_detector
from face_recognition_ros.third_party import mtcnn_mxnet, align_mtcnn
from face_recognition_ros.utils import files


class MtcnnMxnetDetector(base_face_detector.BaseFaceDetector):
    def __init__(self, conf):
        self.detector = mtcnn_mxnet.MtcnnDetector(
            model_folder=files.PROJECT_ROOT + "/data/models/mtcnn-mxnet"
        )

    def extract_region(self, image, threshold=0.0):
        ret = self.detector.detect_face(image)

        if ret is None:
            return [], None

        bbox, points = ret
        if bbox.shape[0] == 0:
            return [], None

        regions = [
            region.RectangleRegion(bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1], bb[4])
            for bb in bbox
            if bb[4] >= threshold
        ]

        return regions, (bbox, points)

    def extract_images(self, image, regions=None, raw_detection=None, align=True):
        if raw_detection is None and regions is None:
            regions, raw_detection = self.extract_region(image, 0)

        if not align:
            return super(MtcnnMxnetDetector, self).extract_images(
                image, regions, raw_detection
            )
        else:
            res = []
            if raw_detection is not None:
                bbox, points = raw_detection

                for idx, box in enumerate(bbox):
                    point = points[idx, :].reshape((2, 5)).T
                    aligned = align_mtcnn.align(image, box, point, image_size="112,112")
                    res.append(aligned)

            return res
