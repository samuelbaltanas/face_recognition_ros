import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn import neighbors, preprocessing, svm


class KNNMatcher:
    def __init__(self, config):
        with open(
            "/home/sam/UMA/4/4_2/3-TFG/3-Workspace/face_recognition_ros/face_recognition_ros/data/database/family_knn/label_encoder.pickle"
        ) as f:
            self.enc = pickle.load(f)  # type: preprocessing.LabelEncoder
        with open(
            "/home/sam/UMA/4/4_2/3-TFG/3-Workspace/face_recognition_ros/face_recognition_ros/data/database/family_knn/classifier.pickle"
        ) as f:
            self.classif = pickle.load(f)  # type: svm.SVC

    def recognize(self, embedding, threshold=0.6):
        embedding = embedding / np.linalg.norm(embedding, axis=1)[:, None]
        pred = self.classif.predict_proba(embedding)[0]
        clas = np.argmax(pred)
        d = pred[clas]
        if d >= threshold:
            iden = self.enc.inverse_transform([clas])[0]
            logging.info("Id: {} recognized with score {}".format(iden, d))
            return iden, d
        else:
            logging.debug(
                "Unknown face. Nearest: [{}] Score={}".format(
                    self.enc.inverse_transform([clas])[0], d
                )
            )
            return "", d


def create_knn_classifier(pandas_pickle, out_dir):

    df = pd.read_pickle(pandas_pickle)
    labels = df.identities.unique()

    lab_enc = preprocessing.LabelEncoder()
    lab_enc.fit(labels)

    X = np.vstack(df.embeddings.values)
    X = X / np.linalg.norm(X, axis=1)[:, None]
    Y = lab_enc.transform(df.identities)

    clf = neighbors.KNeighborsClassifier(
        5
    )  # "cosine")  # , weights="distance")
    clf.fit(X, Y)

    with open(os.path.join(out_dir, "classifier.pickle"), "w") as f:
        f.write(pickle.dumps(clf))
    with open(os.path.join(out_dir, "label_encoder.pickle"), "w") as f:
        f.write(pickle.dumps(lab_enc))


if __name__ == "__main__":
    create_knn_classifier(
        "/home/sam/UMA/4/4_2/3-TFG/3-Workspace/face_recognition_ros/face_recognition_ros/data/database/family_dataset/database.pkl",
        "/home/sam/UMA/4/4_2/3-TFG/3-Workspace/face_recognition_ros/face_recognition_ros/data/database/family_knn",
    )
