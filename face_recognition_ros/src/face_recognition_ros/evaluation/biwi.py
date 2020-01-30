# https://github.com/natanielruiz/deep-head-pose/blob/master/code/datasets.py
# https://github.com/shamangary/FSA-Net/blob/master/demo/demo_FSANET.py

import os
from os import path

import numpy as np
import cv2

BIWI_DIR = "/home/sam/datasets/hpdb/"
ANNOT_DIR = "/home/sam/datasets/hpdb/"
# ANNOT_DIR = "/home/sam/datasets/hpdb/db_annotations/"

rvec = np.array([[517.679, 0, 320], [0, 517.679, 240.5], [0, 0, 1]])

cameraMatrix = np.eye(3)

tvec = np.zeros((3, 1))

dist_coeffs = (0, 0, 0, 0)


def projectPoints(points):
    pp, _ = cv2.projectPoints(points, rvec, tvec, cameraMatrix, dist_coeffs)
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

