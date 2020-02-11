import typing

import numpy as np

from face_recognition_ros import region as reg


class Datum:
    def __init__(
        self,
        region=None,
        image=None,
        keypoints=None,
        embedding=None,
        identity=None,
        match_score=4.0,
    ):
        # DETECTION
        self.region = region  # type: typing.Optional[reg.RectangleRegion]
        self.image = image  # type: typing.Optional[np.ndarray]
        self.keypoints = keypoints
        # RECOGNITION
        self.embedding = embedding  # type: typing.Optional[np.ndarray]
        # MATCHING
        self.identity = identity  # type: typing.Optional[str]
        self.match_score = match_score  # type: float

    def __str__(self):
        return "\n".join(
            [
                # "Pose: {}".format(self.pose),
                # "Pose score: {}".format(self.pose_score),
                "Region: {}".format(self.region),
                # "Image: {}".format(self.face_image),
                # "Embedding: {}".format(self.embedding),
                "Identity: {}".format(self.identity),
                "Match score: {}".format(self.match_score),
            ]
        )

    def draw(self, image, **kwargs):
        # type: (np.ndarray) -> np.ndarray
        if self.region is not None:
            image = self.region.draw(image, label=self.identity, **kwargs)

        return image
