import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn import preprocessing, svm


class SVMMatcher:
    def __init__(self, config):
        with open(
            "/home/sam/UMA/4/4_2/3-TFG/3-Workspace/face_recognition_ros/face_recognition_ros/data/database/family_svm/label_encoder.pickle"
        ) as f:
            self.enc = pickle.load(f)  # type: preprocessing.LabelEncoder
        with open(
            "/home/sam/UMA/4/4_2/3-TFG/3-Workspace/face_recognition_ros/face_recognition_ros/data/database/family_svm/classifier.pickle"
        ) as f:
            self.classif = pickle.load(f)  # type: svm.SVC

    def recognize(self, embedding, threshold=0.50):
        embedding = embedding / np.linalg.norm(embedding, axis=1)[:, None]
        pred = self.classif.predict_proba(embedding)[0]
        clas = np.argmax(pred)
        d = pred[clas]
        if d >= threshold:
            iden = self.enc.inverse_transform([clas])[0]
            logging.info("Id: {} recognized with score {}".format(iden, d))

            # TODO: Pipe into verifier

            return iden, d
        else:
            logging.debug(
                "Unknown face. Nearest: [{}] Score={}".format(
                    self.enc.inverse_transform([clas])[0], d
                )
            )
            return "", d


def create_svm_classifier(pandas_picke, out_dir):
    df = pd.read_pickle(pandas_picke)
    labels = df.identities.unique()

    lab_enc = preprocessing.LabelEncoder()
    lab_enc.fit(labels)

    X = np.vstack(df.embeddings.values)
    X = X / np.linalg.norm(X, axis=1)[:, None]
    Y = lab_enc.transform(df.identities)

    clf = svm.SVC(C=1.0, kernel="linear", probability=True)
    clf.fit(X, Y)

    with open(os.path.join(out_dir, "classifier.pickle"), "w") as f:
        f.write(pickle.dumps(clf))
    with open(os.path.join(out_dir, "label_encoder.pickle"), "w") as f:
        f.write(pickle.dumps(lab_enc))


if __name__ == "__main__":
    create_svm_classifier(
        "/home/sam/UMA/4/4_2/3-TFG/3-Workspace/face_recognition_ros/face_recognition_ros/data/database/family_dataset/dataset_ark.pkl",
        "/home/sam/UMA/4/4_2/3-TFG/3-Workspace/face_recognition_ros/face_recognition_ros/data/database/family_svm",
    )
