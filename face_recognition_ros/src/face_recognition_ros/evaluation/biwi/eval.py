import itertools
from os import path

import cv2
import pandas as pd
from tqdm.autonotebook import tqdm, trange

from face_recognition_ros import detection, encoding_arc, recognition
from face_recognition_ros.evaluation.biwi import aux, biwi
from face_recognition_ros.utils import config


def create_biwi_db(csv_path: str, out_path: str):
    dataset = biwi.BiwiDataset()

    # Processing pipeline
    config.load_config()
    # config.logger_config()
    detector = detection.FaceDetector()
    encoder = encoding_arc.FaceEncoder()

    labels = []
    embeddings = []

    aux_info = biwi.load_identities(csv_path)

    # Create faces database (MIN_ANGLES)
    for idx in trange(len(aux_info)):

        if idx + 1 in aux.REPEATED:
            continue

        label = aux_info.iden[idx]
        if label in labels:
            continue

        iden = aux_info.folder[idx]
        frame = aux_info.frame[idx]
        # im_path, (center3D, angle) = dataset[int(iden), frame]
        dat = dataset[int(iden), frame]

        image = dat.read_image()

        data = detector.predict(image, extract_image=True, align=True)
        face_match = biwi.match_detection(data, (dat.center3d, dat.angle))
        embedding = encoder.predict([data[face_match].image])

        labels.append(label)
        embeddings.append(embedding[0])

    df = pd.DataFrame({"identities": labels, "embeddings": embeddings})
    df.to_pickle(out_path)
    # logging.info("Face embeddings saved to {}".format(out_path))
    return df


def eval_on_biwi(store_file, results_fol, aux_path=None):
    config.load_config()
    config.CONFIG["STORAGE"]["database_file"] = store_file

    aux_info = biwi.load_identities(aux_path)
    face_rec = recognition.Recognition()
    dataset = biwi.BiwiDataset()

    start = 0
    results = {
        "image_id": [],
        "image_frame": [],
        "score": [],
        "pred_id": [],
        "true_id": [],
        "roll": [],
        "yaw": [],
        "pitch": [],
    }

    with tqdm(total=len(dataset)) as pbar:
        for ctr, dat in itertools.islice(enumerate(dataset), start, None):
            image = cv2.imread(dat.path)
            faces = face_rec.recognize(image)
            match = biwi.match_detection(faces, (dat.center3d, dat.angle))

            if match is None:
                results["score"].append(1.0)
                results["pred_id"].append("???")
            else:
                results["score"].append(faces[match].match_score)
                results["pred_id"].append(faces[match].identity)
            results["true_id"].append(aux_info.iden[dat.identity - 1])

            results["image_id"].append(dat.identity)
            results["image_frame"].append(dat.frame)

            results["roll"].append(dat.angle[0])
            results["pitch"].append(dat.angle[1])
            results["yaw"].append(dat.angle[2])

            pbar.update(1)

            data = pd.DataFrame(results)
            data.to_pickle(path.join(results_fol, "results_{}.pkl".format(ctr)))
            results = {
                "image_id": [],
                "image_frame": [],
                "score": [],
                "pred_id": [],
                "true_id": [],
                "roll": [],
                "yaw": [],
                "pitch": [],
            }

    data = pd.DataFrame(results)
    data.to_pickle(path.join(results_fol, "results.pkl"))
