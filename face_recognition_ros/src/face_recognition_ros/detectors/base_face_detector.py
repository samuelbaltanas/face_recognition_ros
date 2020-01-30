import numpy as np
import typing

from face_recognition_ros import region


class BaseFaceDetector(object):
    def __init__(self):
        raise NotImplementedError()

    def extract_region(self, image):
        # type: (np.ndarray)
        raise NotImplementedError()

    def extract_images(self, image, regions=None, raw_detection=None):
        # type: (np.ndarray, typing.Optional[region.Region], typing.Any) -> typing.List[np.ndarray]
        if regions is None:
            regions, _ = self.extract_region(image)
        return [r.extract_face(image) for r in regions]
