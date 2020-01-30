# https://github.com/natanielruiz/deep-head-pose/blob/master/code/datasets.py
# https://github.com/shamangary/FSA-Net/blob/master/demo/demo_FSANET.py

import os
from os import path
import typing

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import cv2

from face_recognition_ros import detection, encoding_arc, recognition
from face_recognition_ros import datum
from face_recognition_ros.utils import config

# Default paths to the dataset
BIWI_DIR = "/home/sam/datasets/hpdb/"
ANNOT_DIR = "/home/sam/datasets/hpdb/"
# ANNOT_DIR = "/home/sam/datasets/hpdb/db_annotations/"

# Configuration of the depth camera in the BIWI dataset
RVEC = np.array([[517.679, 0, 320], [0, 517.679, 240.5], [0, 0, 1]])
CAMERA_MATRIX = np.eye(3)
TVEC = np.zeros((3, 1))
DIST_COEFFS = (0, 0, 0, 0)

MIN_ANGLES = [
    269,
    504,
    395,
    476,
    493,
    431,
    390,
    29,
    616,
    425,
    382,
    276,
    216,
    792,
    389,
    503,
    138,
    207,
    129,
    198,
    255,
    280,
    327,
    192,
]


def seek_min_angles():
    min_frame = np.full(24, -1)
    min_angles = np.full(24, np.infty)
    dataset = BiwiDataset()

    for iden, frame in dataset:
        image_path, (center3D, angle) = dataset[iden, frame]
        idx = int(iden) - 1
        abs_angle = np.abs(angle[0]) + np.abs(angle[1]) + np.abs(angle[2])

        if min_angles[idx] > abs_angle:
            min_frame[idx] = frame
            min_angles[idx] = abs_angle

    return min_frame


def create_biwi_db(out_path):
    dataset = BiwiDataset()

    # Processing pipeline
    config.load_config()
    # config.logger_config()
    detector = detection.FacialDetector()
    encoder = encoding_arc.EncodingArc()

    labels = []
    embeddings = []

    # Create faces database (MIN_ANGLES)
    for idx, frame in enumerate(MIN_ANGLES):
        iden = idx + 1
        im_path, (center3D, angle) = dataset[iden, frame]

        image = cv2.imread(im_path)

        x = detector.extract_region(image)
        face_match = match_detection(x, (center3D, angle))
        region, (bbox, points) = x
        face = detector.extract_images(
            image,
            [region[face_match]],
            (bbox[[face_match]], points[[face_match]]),
        )
        embedding = encoder.predict(face)

        labels.append(str(iden))
        embeddings.append(embedding[0])

    df = pd.DataFrame({"identities": labels, "embeddings": embeddings})
    df.to_pickle(out_path)
    # logging.info("Face embeddings saved to {}".format(out_path))
    return df


def eval_on_biwi(store_file, results_file):
    config.load_config()
    config.CONFIG["STORAGE"]["database_file"] = store_file
    face_rec = recognition.Recognition()
    dataset = BiwiDataset()

    results = {
        "is_same": [],
        "score": [],
        "label": [],
        "roll": [],
        "yaw": [],
        "pitch": [],
    }

    # Eval
    for ctr, (iden, frame) in enumerate(dataset):
        image_path, (center3D, angle) = dataset[iden, frame]
        image = cv2.imread(image_path)
        faces = face_rec.recognize(image)  # type: typing.List[datum.Datum]
        match = match_detection((i.region for i in faces), (center3D, angle))

        face = faces[match]
        if match is None:
            results["score"].append(np.inf)
        else:
            results["score"].append(face.match_score)
        results["is_same"].append(face.identity == iden)
        results["label"].append(iden)

        results["roll"].append(angle[0])
        results["pitch"].append(angle[1])
        results["yaw"].append(angle[2])

        if ctr % 10 == 0:
            print("Iter: {}".format(ctr))

    df = pd.DataFrame(results)
    df.to_pickle(results_file)
    return df


def projectPoints(points):
    pp, _ = cv2.projectPoints(points, RVEC, TVEC, CAMERA_MATRIX, DIST_COEFFS)
    return pp


def read_ground_truth(fd):
    mat = []
    with open(fd, "r") as fp:
        for line in fp:
            ll = []
            items = line.split()
            if len(items) == 0:
                continue
            for i in items:
                ll.append(float(i))
            mat.append(ll)

    m = np.array(mat)
    translation = m[3, :]
    rotation = m[:3, :]

    return translation, angle_from_matrix(rotation)


def angle_from_matrix(rotation):
    r = rotation.T

    roll = -np.arctan2(r[1, 0], r[0][0]) * 180 / np.pi
    yaw = (
        -np.arctan2(-r[2, 0], np.sqrt(r[2, 1] ** 2 + r[2, 2] ** 2))
        * 180
        / np.pi
    )
    pitch = np.arctan2(r[2, 1], r[2, 2]) * 180 / np.pi

    return (roll, pitch, yaw)


def draw_axis(img, angle, center=None, size=40):

    roll, pitch, yaw = angle

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if center is None:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2
    else:
        tdx = center[0]
        tdy = center[1]

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    # X-Axis pointing to right. drawn in red
    x1 = size * (cy * cr) + tdx
    y1 = size * (cp * sr + cr * sp * sy) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cy * sr) + tdx
    y2 = size * (cp * cr - sp * sy * sr) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sy) + tdx
    y3 = size * (-cy * sp) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    return img


def draw_biwi_image(image, det_regions, ground_truth, match=None, iden=""):
    plt.figure()
    center, angle = ground_truth
    reg, (bb, pt) = det_regions

    for idx, i in enumerate(reg):
        if idx == match:
            i.draw(image, label=str(iden))
        else:
            i.draw(image)
    if match is not None:
        pt = pt[match].reshape((2, 5)).T
        center2D = pt[2]
        draw_axis(image, angle, center2D)
    plt.imshow(image)
    pp = projectPoints(center)[0].ravel()
    plt.scatter(pp[0], pp[1], color="magenta")
    plt.show()


def match_detection(det_regions, ground_truth):
    if len(det_regions[0]) == 0:
        return None

    center, angle = ground_truth
    pp = projectPoints(center)[0].ravel()

    best = (None, np.infty)
    for idx, region in enumerate(det_regions[0]):
        og_x = region.origin[0, 0]
        og_y = region.origin[1, 0]
        d_x = region.dimensions[0, 0]
        d_y = region.dimensions[1, 0]

        if (
            pp[0] < og_x
            or pp[1] < og_y
            or pp[0] > og_x + d_x
            or pp[1 > og_y + d_y]
        ):
            continue
        else:
            center_bb = np.array((og_x + d_x / 2, og_y + d_y / 2))
            dist = np.sqrt(
                (pp[0] - center_bb[0]) ** 2 + (pp[1] - center_bb[1]) ** 2
            )
            if dist < best[1]:
                best = idx, dist
    return best[0]


class BiwiDataset:
    def __init__(self, path=BIWI_DIR):
        self._dir = path

    def __getitem__(self, key):
        individual, image = key
        folder = path.join(self._dir, "{:02d}".format(individual))

        pose_file = path.join(folder, "frame_{:05d}_pose.txt".format(image))
        image_file = path.join(folder, "frame_{:05d}_rgb.png".format(image))

        if not path.isfile(image_file):
            raise IndexError("File {} does not exist.".format(image_file))

        return image_file, read_ground_truth(pose_file)

    def __iter__(self):
        for identifier in range(1, 25):
            f1 = "{:02d}".format(identifier)
            parent = path.join(self._dir, f1)
            # Gather individual images
            for f in os.scandir(parent):
                f = f.name
                if f.endswith(".png"):
                    frame = int(f.split("_")[1])
                    yield identifier, frame


if __name__ == "__main__":
    DB_PATH = "/home/sam/Desktop/biwi.pkl"
    RES_PATH = "/home/sam/Desktop/biwi_res.pkl"
    config.load_config()
    df = create_biwi_db(DB_PATH)
