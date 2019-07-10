""" Collection of data structures to define regions within an image """
import numpy as np
import cv2

from face_recognition_ros.utils.math import rotate


class RectangleRegion:
    def __init__(
        self, left_x=0.0, top_y=0.0, width=0.0, height=0.0, detection_score=0.0
    ):
        self.origin = np.array([[left_x], [top_y]], dtype=float)
        self.dimensions = np.array([[width], [height]], dtype=float)
        self.detection_score = float(detection_score)

    def extract_face(self, image, shape=None):
        if shape is None:
            shape = (160, 160)

        face = image[
            int(self.origin[0, 0]) : int(
                self.origin[0, 0] + self.dimensions[0, 0]
            ),
            int(self.origin[1, 0]) : int(
                self.origin[1, 0] + self.dimensions[1, 0]
            ),
        ]
        face = cv2.resize(face, shape, interpolation=cv2.INTER_AREA)

        return face

    def to_bounding_box(self):
        res = np.tile(self.origin.T, (4, 1))  # [topl, topr, botl, botr]

        res[1, 0] += self.dimensions[0, 0]
        res[3, 1] += self.dimensions[1, 0]
        res[2] += self.dimensions[:, 0]

        return res

    def draw(self, image, color=(0, 255, 255), thickness=5):
        bb = self.to_bounding_box()
        bb = bb.reshape((-1, 1, 2))
        image = cv2.polylines(image, np.int64([bb]), True, color, thickness)
        return image

    def __repr__(self):
        return "{} {} {} {} {}".format(
            self.origin[0, 0],
            self.origin[1, 0],
            self.dimensions[0, 0],
            self.dimensions[1, 0],
            self.detection_score,
        )


class EllipseRegion:
    def __init__(
        self,
        major_axis_radius=0.0,
        minor_axis_radius=0.0,
        angle=0.0,
        center_x=0.0,
        center_y=0.0,
        detection_score=0.0,
    ):
        if major_axis_radius >= minor_axis_radius:
            self.axis_radius = np.array(
                [[major_axis_radius], [minor_axis_radius]], dtype=float
            )
            self.angle = 0.0
        else:
            self.axis_radius = np.array(
                [[minor_axis_radius], [major_axis_radius]], dtype=float
            )
            self.angle = np.pi / 2

        self.center = np.array([[center_x], [center_y]], dtype=float)
        self.angle += float(angle)
        self.detection_score = float(detection_score)

    def to_bounding_box(self):
        res = np.tile(self.center.T, (4, 1))  # [topl, topr, botl, botr]

        true_axis = rotate(self.axis_radius, self.angle, self.center)
        res = res + (
            np.tile(true_axis.T, (4, 1)) * [[-1, -1], [1, -1], [-1, 1], [1, 1]]
        )

        return res

    def draw(self, image, color=(0, 255, 255), thickness=5):
        return cv2.ellipse(
            image,
            (int(self.center[0, 0]), int(self.center[1, 0])),
            (int(self.axis_radius[0, 0]), int(self.axis_radius[1, 0])),
            angle=self.angle * 180 / np.pi,
            startAngle=0,
            endAngle=360,
            color=color,
            thickness=thickness,
        )

    def extract_face(self, image, shape=None):
        if shape is None:
            shape = (160, 160)

        fitted = np.float32(
            [[0, 0], [shape[0], 0], [0, shape[1]], [shape[0], shape[1]]]
        )
        M = cv2.getPerspectiveTransform(self.to_bounding_box(), fitted)
        face = cv2.warpPerspective(image, M, shape)

        return face

    def __repr__(self):
        return "{} {} {} {} {} {}".format(
            self.axis_radius[0, 0],
            self.axis_radius[1, 0],
            self.angle,
            self.center[0, 0],
            self.center[1, 0],
            self.detection_score,
        )
