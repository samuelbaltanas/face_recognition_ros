from __future__ import print_function

from os import path

import numpy as np

from face_recognition_ros import storage


class FaceMatcher():

    def __init__(self, storage_config):
        # type: (dict) -> None
        self.database = storage.FaceDatabase()
        self.database.load(
            path.join(
                storage_config["database_folder"],
                storage_config["database_name"]
            )
        )

    def recognize(self, embeding):
        # type: (np.ndarray) -> (str, float)
        dist = np.sqrt(
            np.sum((self.database.embeddings - embeding)**2, axis=0)
        )  # type: np.ndarray

        idx = dist.argmin(axis=0)

        return self.database[idx], dist[idx]
