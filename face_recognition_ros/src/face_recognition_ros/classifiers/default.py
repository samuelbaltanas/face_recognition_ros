import logging

import numpy as np
import pickle
import pandas as pd

from face_recognition_ros.utils import config
from face_recognition_ros.utils.math import dist


class FaceMatcher:
    def __init__(self, conf=None):
        # type: (dict) -> None
        if conf is None:
            conf = config.CONFIG
        storage_config = conf["STORAGE"]
        with open(storage_config["database_file"]) as f:
            database = pickle.load(f)  # type: pd.DataFrame
            self.labels = database.identities.to_numpy(copy=True)
            self.embeddings = np.vstack(database.embeddings.array)
            del database

    def recognize1(self, embeding, threshold=1.1):
        # type: (np.ndarray) -> (str, float)

        d = dist(self.embeddings, embeding, func=0)  # type: np.ndarray
        res = []
        for label, df in self.database.groupby(by="identities"):
            # d = np.sqrt(np.sum((embs - embeding) ** 2, axis=1))  # type: np.ndarray

            m = np.mean(d <= threshold)

            if m > 0:
                logging.debug("Hypothesis [{}] accepted. Distance={}".format(label, m))
                res.append((label, m))
            else:
                logging.debug("Hypothesis [{}] discarded. Distance={}".format(label, m))

        if not res:
            logging.debug("Unknown face. Distance={}".format(np.NaN))
            return "", np.NaN
        else:
            label, d = max(res, key=lambda x: x[1])
            logging.info("Id: {} recognized with dist {}".format(label, d))
            return label, d

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
