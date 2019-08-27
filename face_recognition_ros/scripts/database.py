import os
import logging

import cv2
import pandas as pd

from face_recognition_ros import detection, encoding_arc
from face_recognition_ros.utils import config, files


def create_faces_dataset(in_dir, out_dir=None, out_file="database.pkl"):
    # Path fixing
    if out_dir is None:
        out_dir = in_dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Processing pipeline
    config.load_config()
    config.logger_config()
    det = detection.FacialDetector()
    enc = encoding_arc.EncodingArc()

    labels = []
    embeddigs = []

    for label, file_list in files.image_folder_traversal(in_dir):
        logging.info("Adding {} to database".format(label))
        for file_name in file_list:
            image = cv2.imread(file_name)
            if image is None:
                continue
            try:
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = det.extract_images(image)
                embedding = enc.predict(faces)
                if embedding.shape[0] > 1:
                    logging.warn("Multiple faces in image image {}".format(file_name))
                    continue

            except Exception as e:
                logging.warn("Error in image {}".format(file_name))
                logging.warn(e.message)
                # os.remove(file_name)
                # logging.warn("File {} deleted.".format(file_name))
            else:
                logging.debug("Image added: {}".format(file_name))
                labels.append(label)
                embeddigs.append(embedding[0])
    df = pd.DataFrame({"identities": labels, "embeddings": embeddigs})
    
    out_path = os.path.join(out_dir, out_file)
    df.to_pickle(out_path)
    logging.info("Face embeddings saved to {}".format(out_path))


if __name__ == "__main__":
    create_faces_dataset(
        "/home/sam/UMA/4/4_2/3-TFG/3-Workspace/face_recognition_ros/face_recognition_ros/data/database/family_dataset",
        out_file="dataset_ark_mobile.pkl",
    )
