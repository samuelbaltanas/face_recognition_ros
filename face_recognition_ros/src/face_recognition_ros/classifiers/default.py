import logging

import numpy as np
import pandas as pd

from face_recognition_ros.utils import config
from face_recognition_ros.utils.math import dist


class FaceMatcher:
    def __init__(self, conf=None):
        # type: (dict) -> None
        if conf is None:
            conf = config.CONFIG
        storage_config = conf["STORAGE"]
        database = pd.read_pickle(
            storage_config["database_file"]
        )  # type: pd.DataFrame
        self.labels = database.identities.to_numpy(copy=True, dtype=str)
        self.embeddings = np.vstack(database.embeddings.array)
        del database

    def recognize(self, embeding, threshold=0.4):
        # type: (np.ndarray) -> (str, float)
        d_list = dist(self.embeddings, embeding, func=1)  # type: np.ndarray

        idx = d_list.argmin(axis=0)
        d = d_list[idx]

        if d_list[idx] < threshold:
            iden = self.labels[idx]
            logging.info("Id: {} recognized with dist {}".format(iden, d))
            return iden, d
        else:
            logging.debug("Unknown face. Distance={}".format(d))
            return "", d
