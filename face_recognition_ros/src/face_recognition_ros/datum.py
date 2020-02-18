import typing

import numpy as np

from face_recognition_ros import region as reg


class Datum:
    def __init__(
        self,
        region: typing.Optional[reg.BoundingBox] = None,
        image: typing.Optional[np.ndarray] = None,
        keypoints=None,
        embedding: typing.Optional[np.ndarray] = None,
        identity: typing.Optional[str] = None,
        match_score=4.0,
    ):
        # DETECTION
        self.region = region
        self.image = image
        self.keypoints = keypoints
        # RECOGNITION
        self.embedding = embedding
        # MATCHING
        self.identity = identity
        self.match_score = match_score

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
