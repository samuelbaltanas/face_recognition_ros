# https://github.com/natanielruiz/deep-head-pose/blob/master/code/datasets.py
# https://github.com/shamangary/FSA-Net/blob/master/demo/demo_FSANET.py

import itertools
import os
import typing
from os import path

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm, trange

from face_recognition_ros import datum, detection, encoding_arc, recognition
from face_recognition_ros.utils import config

# Default paths to the dataset
BIWI_DIR = "/home/sam/Datasets/data/hpdb/"

# Configuration of the depth camera in the BIWI dataset
RVEC = np.array([[517.679, 0, 320], [0, 517.679, 240.5], [0, 0, 1]])
CAMERA_MATRIX = np.eye(3)
TVEC = np.zeros((3, 1))
DIST_COEFFS = (0, 0, 0, 0)

iden = np.array(
    [
        "F01",
        "F02",
        "F03",
        "F04",
        "F05",
        "F06",
        "M01",
        "M02",
        "M03",
        "M04",
        "M05",
        "M06",
        "M07",
        "M08",
        "F03",
        "M09",
        "M10",
        "F05",
        "M11",
        "M12",
        "F02",
        "M01",
        "M13",
        "M14",
    ],
    dtype=str,
)
sex = np.array(
    [
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "M",
        "M",
        "M",
        "M",
        "M",
        "M",
        "M",
        "M",
        "M",
        "M",
        "M",
        "F",
        "M",
        "M",
        "F",
        "M",
        "M",
        "M",
    ],
    dtype=str,
)

repeated = {
    15: 3, 18: 5, 21: 2, 22: 7
    }


class Angle(typing.NamedTuple):
    roll: float
    pitch: float
    yaw: float


class BiwiDatum(typing.NamedTuple):
    path: str
    center3d: np.ndarray
    angle: Angle


def load_identities(path=path.join(BIWI_DIR, "dataset_centered.csv")):
    cinv = {"folder": str, "iden": str, "frame": int, "sex": str}
    return pd.read_csv(path, converters=cinv)


def seek_min_angles():
    min_frame = np.full(24, -1)
    min_angles = np.full(24, np.infty)
    dataset = BiwiDataset()

    for iden, frame in dataset:
        angle = dataset[iden, frame].angle
        idx = int(iden) - 1
        abs_angle = np.abs(angle[0]) + np.abs(angle[1]) + np.abs(angle[2])

        if min_angles[idx] > abs_angle:
            min_frame[idx] = frame
            min_angles[idx] = abs_angle

    return min_frame


def create_biwi_db(csv_path: str, out_path: str):
    dataset = BiwiDataset()

    # Processing pipeline
    config.load_config()
    # config.logger_config()
    detector = detection.FaceDetector()
    encoder = encoding_arc.FaceEncoder()

    labels = []
    embeddings = []

    aux_info = load_identities(csv_path)

    # Create faces database (MIN_ANGLES)
    for idx in trange(len(aux_info)):
        label = aux_info.iden[idx]
        if label in labels:
            continue

        iden = aux_info.folder[idx]
        frame = aux_info.frame[idx]
        # im_path, (center3D, angle) = dataset[int(iden), frame]
        dat = dataset[int(iden), frame]

        image = cv2.imread(dat.path)

        data = detector.predict(image, extract_image=True, align=True)
        face_match = match_detection(data, (dat.center3d, dat.angle))
        embedding = encoder.predict([data[face_match].image])

        labels.append(label)
        embeddings.append(embedding[0])

        if idx % 5 == 4:
            print("\rProgress: {}/{}".format(idx + 1, len(aux_info)), end="")

    df = pd.DataFrame({"identities": labels, "embeddings": embeddings})
    df.to_pickle(out_path)
    print("\rProgress: {}/{}".format(idx + 1, len(aux_info)), end="")
    # logging.info("Face embeddings saved to {}".format(out_path))
    return df


def eval_on_biwi(store_file, results_fol, store_each=-1, overwrite=False):
    config.load_config()
    config.CONFIG["STORAGE"]["database_file"] = store_file

    aux_info = load_identities()
    face_rec = recognition.Recognition()
    dataset = BiwiDataset()

    cached_results = len(os.listdir(results_fol))
    start = int(store_each * cached_results) if store_each > 0 else 0
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

    with tqdm(total=15678) as pbar:
        # Eval
        for ctr, (iden, frame) in itertools.islice(
            enumerate(dataset), start, None
        ):
            dat = dataset[iden, frame]
            image = cv2.imread(dat.path)
            faces = face_rec.recognize(image)  # type: typing.List[datum.Datum]
            match = match_detection(faces, (dat.center3d, dat.angle))

            if match is None:
                results["score"].append(1.0)
                results["pred_id"].append("???")
            else:
                results["score"].append(faces[match].match_score)
                results["pred_id"].append(faces[match].identity)
            results["true_id"].append(aux_info.iden[iden - 1])

            results["image_id"].append(iden)
            results["image_frame"].append(frame)

            results["roll"].append(dat.angle[0])
            results["pitch"].append(dat.angle[1])
            results["yaw"].append(dat.angle[2])

            pbar.update(1)

            if store_each > 0 and ctr % store_each == store_each - 1:
                df = pd.DataFrame(results)
                df.to_pickle(
                    path.join(results_fol, "results_{}.pkl".format(ctr))
                )
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

    s = "" if store_each <= 0 else "_END"

    df = pd.DataFrame(results)
    df.to_pickle(path.join(results_fol, "results{}.pkl".format(s)))


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

    return Angle(roll, pitch, yaw)


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


def draw_biwi_image(
    image, det_data: typing.List[datum.Datum], ground_truth, match=None, iden=""
):
    plt.figure()
    center, angle = ground_truth
    for idx, dat in enumerate(det_data):
        if idx == match:
            dat.identity = str(iden)
        dat.draw(image)
    if match is not None:
        pt = det_data[match].keypoints.reshape((2, 5)).T
        center2D = pt[2]
        draw_axis(image, angle, center2D)
    plt.imshow(image)
    pp = projectPoints(center)[0].ravel()
    plt.scatter(pp[0], pp[1], color="magenta")
    plt.show()


def match_detection(det_data: datum.Datum, ground_truth: tuple):
    center, _ = ground_truth
    pp = projectPoints(center)[0].ravel()

    best = (None, np.infty)
    for idx, dat in enumerate(det_data):
        bb = dat.region.box[0]

        if pp[0] < bb[0] or pp[1] < bb[1] or pp[0] > bb[2] or pp[1] > bb[3]:
            continue
        else:
            center_bb = np.array((bb[0] + bb[2] / 2, bb[1] + bb[3] / 2))
            dist = np.sqrt(
                (pp[0] - center_bb[0]) ** 2 + (pp[1] - center_bb[1]) ** 2
            )
            if dist < best[1]:
                best = idx, dist
    return best[0]


class BiwiDataset:
    def __init__(self, path=BIWI_DIR):
        self._dir = path

    def __getitem__(self, key: typing.Tuple[int, int]) -> BiwiDatum:
        individual, image = key
        folder = path.join(self._dir, "{:02d}".format(individual))

        pose_file = path.join(folder, "frame_{:05d}_pose.txt".format(image))
        image_file = path.join(folder, "frame_{:05d}_rgb.png".format(image))

        if not path.isfile(image_file):
            raise IndexError("File {} does not exist.".format(image_file))

        center3d, angle = read_ground_truth(pose_file)

        return BiwiDatum(image_file, center3d, angle)

    def __iter__(self):
        for identifier in range(1, 25):
            f1 = "{:02d}".format(identifier)
            parent = path.join(self._dir, f1)
            # Gather individual images
            for f in sorted(
                f.name for f in os.scandir(parent) if f.name.endswith(".png")
            ):
                frame = int(f.split("_")[1])
                yield identifier, frame

    def as_pandas(self):
        d = {
            "identity": [],
            "frame": [],
            "center_x": [],
            "center_y": [],
            "center_z": [],
            "roll": [],
            "pitch": [],
            "yaw": [],
        }
        with tqdm(total=self.__len__) as tq:
            for iden, frame in self:
                dat = self[iden, frame]
                d["identity"].append(iden)
                d["frame"].append(frame)
                d["center_x"].append(dat.center3d[0])
                d["center_y"].append(dat.center3d[1])
                d["center_z"].append(dat.center3d[2])
                d["roll"].append(dat.angle.roll)
                d["pitch"].append(dat.angle.pitch)
                d["yaw"].append(dat.angle.yaw)
                tq.update(1)

        return pd.DataFrame(d)

    def __len__(self):
        return 15678


if __name__ == "__main__":
    DB_PATH = "/home/sam/Desktop/biwi.pkl"
    RES_PATH = "/home/sam/Desktop/biwi_res/"
    config.load_config()
    # df = create_biwi_db(DB_PATH)
    eval_on_biwi(DB_PATH, RES_PATH, store_each=100)
