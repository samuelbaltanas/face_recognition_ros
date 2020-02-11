# coding: utf-8
import numpy as np

from face_recognition_ros import region, datum

# from face_recognition_ros.detectors import base_face_detector
from face_recognition_ros.third_party import mtcnn_mxnet, align_mtcnn
from face_recognition_ros.utils import files


class MtcnnMxnetDetector:
    def __init__(self, conf):
        self.detector = mtcnn_mxnet.MtcnnDetector(
            model_folder=files.PROJECT_ROOT + "/data/models/mtcnn-mxnet"
        )

    def predict(
        self, X: np.ndarray, threshold=0.95, extract_image=False
    ) -> datum.Datum:
        ret = self.detector.detect_face(X)

        if ret is None:
            return []

        bbox, points = ret
        if bbox.shape[0] == 0:
            return []

        data = []
        for bb, point in zip(*ret):

            if bb[4] >= threshold:
                dat = datum.Datum(
                    region=region.RectangleRegion(
                        bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1], bb[4]
                    ),
                    keypoints=point,
                )

                if extract_image:
                    point = point.reshape((2, 5)).T
                    aligned = align_mtcnn.align(
                        X, bb, point, image_size="112,112"
                    )
                    dat.image = aligned
                data.append(dat)

        return data


"""
    def extract_region(self, image: np.ndarray, threshold=0.95) -> datum.Datum:
        ret = self.detector.detect_face(image)

        if ret is None:
            return [], None

        bbox, points = ret
        if bbox.shape[0] == 0:
            return [], None

        data = [
            datum.Datum(
                region=region.RectangleRegion(
                    bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1], bb[4]
                ),
                keypoints=points[idx, :],
            )
            for idx, bb in enumerate(bbox)
            if bb[4] >= threshold
        ]

        return data

    def extract_images(self, image, data=None, align=True):
        if data is None:
            data = self.extract_region(image, 0)

        if not align:
            return super(MtcnnMxnetDetector, self).extract_images(image, data)
        else:
            res = []

            for dat in data:
                point = dat.keypoints.reshape((2, 5)).T
                aligned = align_mtcnn.align(
                    image,
                    dat.region.to_cvbox(score=True),
                    point,
                    image_size="112,112",
                )
                res.append(aligned)

            return res
"""
