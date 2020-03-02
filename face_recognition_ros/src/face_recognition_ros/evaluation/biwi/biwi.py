# https://github.com/natanielruiz/deep-head-pose/blob/master/code/datasets.py
# https://github.com/shamangary/FSA-Net/blob/master/demo/demo_FSANET.py

import os
import typing
from os import path

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm

from face_recognition_ros import datum

# Default paths to the dataset
BIWI_DIR = "/home/sam/Datasets/data/hpdb/"

# Configuration of the depth camera in the BIWI dataset
RVEC = np.array([[517.679, 0, 320], [0, 517.679, 240.5], [0, 0, 1]])
CAMERA_MATRIX = np.eye(3)
TVEC = np.zeros((3, 1))
DIST_COEFFS = (0, 0, 0, 0)


integer = typing.Union[int, np.integer]
Integer = typing.Union[integer, str]


class Angle(typing.NamedTuple):
    """ Wrapper over tuple for use in angles """

    roll: float
    pitch: float
    yaw: float


class BiwiDatum(typing.NamedTuple):
    """ Data wrapper over results. """

    path: str
    center3d: np.ndarray
    angle: Angle
    identity: int
    frame: int

    def read_image(self):
        """ Load frame image. """
        return cv2.imread(self.path)

    def draw(
        self,
        image=None,
        det_data: typing.List[datum.Datum] = None,
        match: typing.Optional[int] = None,
    ):
        if image is None:
            image = self.read_image()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.figure()

        for idx, dat in enumerate(det_data):
            if idx == match:
                dat.identity = str(self.identity)
            dat.draw(image)

        if match is not None:
            keypoints = det_data[match].keypoints.reshape((2, 5)).T
            center_2d = keypoints[2]
            draw_axis(image, self.angle, center_2d)

        plt.imshow(image)
        projected = projectPoints(self.center3d)[0].ravel()
        plt.scatter(projected[0], projected[1], color="magenta")
        plt.show()


def load_identities(aux_path=path.join(BIWI_DIR, "centered.csv")):
    cinv = {"folder": str, "iden": str, "frame": int, "sex": str}
    return pd.read_csv(aux_path, converters=cinv)


def seek_min_angles():
    min_frame = np.full(24, -1)
    min_angles = np.full(24, np.infty)
    dataset = BiwiDataset()

    for frame in dataset:
        angle = frame.angle
        idx = frame.identity - 1
        abs_angle = np.abs(angle[0]) + np.abs(angle[1]) + np.abs(angle[2])

        if min_angles[idx] > abs_angle:
            min_frame[idx] = frame
            min_angles[idx] = abs_angle

    return min_frame


def projectPoints(points):
    """ Convenience method in place of cv2.projectPoints.
        It uses the hardcoded parameters of the BIWI dataset's camera.
    """
    return cv2.projectPoints(points, RVEC, TVEC, CAMERA_MATRIX, DIST_COEFFS)[0]


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


def angle_from_matrix(rotation_matrix: np.ndarray) -> Angle:
    """ Transform rotation matrix to Euclidean angle. """
    rotation_matrix = rotation_matrix.T

    roll = -np.arctan2(rotation_matrix[1, 0], rotation_matrix[0][0]) * 180 / np.pi
    yaw = (
        -np.arctan2(
            -rotation_matrix[2, 0],
            np.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2),
        )
        * 180
        / np.pi
    )
    pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]) * 180 / np.pi

    return Angle(roll, pitch, yaw)


def draw_axis(
    img: np.ndarray, angle: Angle, center=None, arrow_length=40
) -> np.ndarray:
    """ Draw 3D axis using the orientation of the face. """

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
    x1 = arrow_length * (cy * cr) + tdx
    y1 = arrow_length * (cp * sr + cr * sp * sy) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = arrow_length * (-cy * sr) + tdx
    y2 = arrow_length * (cp * cr - sp * sy * sr) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = arrow_length * (sy) + tdx
    y3 = arrow_length * (-cy * sp) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    return img


def match_detection(
    det_data: typing.List[datum.Datum], ground_truth: tuple, margin: float = 0.5
):
    """ Use BIWI's ground truth to find which face detection should be selected.

        :param det_data: List of detections.
        :param ground_truth: Ground truth from BIWI. 3D point corresponding to the center of the face.

    """
    center, _ = ground_truth
    projected = projectPoints(center)[0].ravel()

    best = (None, np.infty)
    for idx, dat in enumerate(det_data):
        bbox = dat.region.box[0].copy()

        diff_x = (bbox[2] - bbox[0]) * margin / 2
        diff_y = (bbox[3] - bbox[1]) * margin / 2

        bbox[0] -= diff_x
        bbox[1] -= diff_y
        bbox[2] += diff_x
        bbox[3] += diff_y

        if (
            projected[0] < bbox[0]
            or projected[1] < bbox[1]
            or projected[0] > bbox[2]
            or projected[1] > bbox[3]
        ):
            continue
        else:
            center_bb = np.array((bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2))
            dist = np.sqrt(
                (projected[0] - center_bb[0]) ** 2 + (projected[1] - center_bb[1]) ** 2
            )
            if dist < best[1]:
                best = idx, dist
    return best[0]


class BiwiIndividual:
    def __init__(self, data_path=BIWI_DIR, index: Integer = 1):
        self.index = index
        self.directory = path.join(data_path, "{:02d}".format(index))

    def __iter__(self):
        # Gather individual's images
        for image_file in sorted(
            fname.name
            for fname in os.scandir(self.directory)
            if fname.name.endswith(".png")
        ):
            index = int(image_file.split("_")[1])
            pose_file = path.join(self.directory, "frame_{}_pose.txt".format(index))
            center3d, angle = read_ground_truth(pose_file)

            yield BiwiDatum(image_file, center3d, angle)

    def __getitem__(self, index: Integer):

        if isinstance(index, (np.integer, int)):
            index = "{:05d}".format(index)

        pose_file = path.join(self.directory, "frame_{}_pose.txt".format(index))
        image_file = path.join(self.directory, "frame_{}_rgb.png".format(index))

        if not path.isfile(image_file):
            raise IndexError("File {} does not exist.".format(image_file))

        center3d, angle = read_ground_truth(pose_file)

        return BiwiDatum(image_file, center3d, angle, self.index, index)


class BiwiDataset:
    """ Class wrapping the BIWI dataset.

        This class is better used as an iterable, as the __getitem__ method is not ensured to succed in the whole range of frames.
        E.g. an individual might contain frames [..., 119, 121, ...].

    """

    def __init__(self, data_path=BIWI_DIR):
        self._dir = data_path

    def __getitem__(
        self, key: typing.Tuple[Integer, ...]
    ) -> typing.Union[BiwiIndividual, BiwiDatum]:

        individual = BiwiIndividual(self._dir, key[0])

        if len(key) == 1:
            res = individual
        else:
            res = individual[key[1]]

        return res

    def __iter__(self):
        for identifier in range(1, 25):
            individual = BiwiIndividual(self._dir, identifier)

            for frame in individual:
                yield frame

    def as_pandas(self):
        """ Return pandas dataframe containing all info in dataset.

            WARN: This operation is costly and should not be called often.
        """
        data = {
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
            for dat in self:
                data["identity"].append(dat.identity)
                data["frame"].append(dat.frame)
                data["center_x"].append(dat.center3d[0])
                data["center_y"].append(dat.center3d[1])
                data["center_z"].append(dat.center3d[2])
                data["roll"].append(dat.angle.roll)
                data["pitch"].append(dat.angle.pitch)
                data["yaw"].append(dat.angle.yaw)
                tq.update(1)

        return pd.DataFrame(data)

    def __len__(self):
        return 15678
