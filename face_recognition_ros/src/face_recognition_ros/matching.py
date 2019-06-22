from __future__ import print_function

from os import path

import numpy as np

from face_recognition_ros import storage
from face_recognition_ros.utils import files


class FaceMatcher:
    def __init__(self, storage_config):
        # type: (dict) -> None
        self.database = storage.FaceDatabase()

        if storage_config["database_folder"] == "":
            data_path = files.get_face_database(
                storage_config["database_name"]
            )
        else:
            data_path = path.join(
                storage_config["database_folder"],
                storage_config["database_name"],
            )
        self.database.load(data_path)

    def recognize(self, embeding):
        # TODO: Decide method of comparison. Maybe use a SVM to decide
        # type: (np.ndarray) -> (str, float)
        dist = np.sqrt(
            np.sum((self.database.embeddings - embeding) ** 2, axis=0)
        )  # type: np.ndarray

        idx = dist.argmin(axis=0)

        return self.database[idx], dist[idx]
