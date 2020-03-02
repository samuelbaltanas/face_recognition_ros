""" Collection of data structures to define regions within an image """
import cv2
import numpy as np


class BoundingBox:
    """Bounding box with format: x0, y0, x1, y1

    """

    def __init__(self, box: np.ndarray, score: float = 0.0):
        if isinstance(box, list) and len(box) == 4:
            box = np.array(box).reshape(1, 4)
        elif isinstance(box, np.ndarray):
            box = box.reshape(1, 4)
        else:
            raise Exception("Expected list or ndarray of lenght 4")

        self.box = box
        self.score = score

    def extract_face(self, image, shape):

        face = image[
            max(self.box[:, 1], 0) : min(self.box[:, 3], image.shape[0]),
            max(self.box[:, 0], 0) : min(self.box[:, 2], image.shape[1]),
        ]
        face = cv2.resize(face, shape)

        return face

    def to_cvbox(self, margin=0.0, score=False):
        cvbox = [
            self.box[:, 0],
            self.box[:, 1],
            self.box[:, 2] - self.box[:, 0],
            self.box[:, 3] - self.box[:, 1],
        ]
        if margin > 0.0:
            cvbox[0] -= cvbox[2] * margin / 2
            cvbox[1] -= cvbox[3] * margin / 2
            cvbox[2] += cvbox[2] * margin / 2
            cvbox[3] += cvbox[3] * margin / 2
        if score:
            cvbox.append(self.score)

        return tuple(cvbox)

    def intersect(self, x):
        ov = (max(self.box[:, 0], x.box[:, 0]) - min(self.box[:, 2], x.box[:, 2])) * (
            max(self.box[:, 1], x.box[:, 1]) - min(self.box[:, 3], self.box[:, 3])
        )

        return ov / (self.size() + x.size() - ov)

    def draw(self, image, label=None, color=(0, 255, 0), thickness=3, font_scale=1):
        b = self.box[0]
        bb = [b[:2], [b[0], b[3]], b[2:], [b[2], b[1]]]

        if label is None or label == "":
            color = (255, 0, 0)

        image = cv2.polylines(
            image, np.int64([bb]), isClosed=True, color=color, thickness=thickness,
        )
        if label:
            image = cv2.putText(
                image,
                label,
                (int(self.box[:, 0]), int(self.box[:, 3] + 30 * font_scale),),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=color,
                thickness=int(thickness * 2 / 3),
                lineType=cv2.FILLED,
            )
        return image

    def __repr__(self):
        return "{}, {}".format(self.box, self.score)

    def is_empty(self):
        return np.any(self.box[:, 2:] == 0.0)

    def size(self):
        return self.box[:, 2] * self.box[:, 3]

    def offset(self, img_center):
        return np.vstack(
            [
                self.box[:, ::2].mean() - img_center[:, 0],
                self.box[:, 1::2].mean() - img_center[:, 1],
            ]
        )


"""
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

    def draw(self, image, label="", color=(0, 255, 255), thickness=3):
        image = cv2.ellipse(
            image,
            (int(self.center[0, 0]), int(self.center[1, 0])),
            (int(self.axis_radius[0, 0]), int(self.axis_radius[1, 0])),
            angle=self.angle * 180 / np.pi,
            startAngle=0,
            endAngle=360,
            color=color,
            thickness=thickness,
        )

        return image

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

    def __str__(self):
        return "\n\t".join(
            [
                "[ellipse]",
                "center: ({}, {})".format(
                    self.center[0, 0], self.center[1, 0]
                ),
                "major_axis_radius: {}".format(self.axis_radius[0, 0]),
                "minor_axis_radius: {}".format(self.axis_radius[1, 0]),
                "angle: {}".format(self.angle),
                "score: {}".format(self.detection_score),
            ]
        )


Region = typing.Union[RectangleRegion, EllipseRegion]
"""
