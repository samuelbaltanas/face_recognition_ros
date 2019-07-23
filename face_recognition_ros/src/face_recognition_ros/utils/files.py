# TODO: Better checks + incorporate config files
import os
from os import path


PROJECT_ROOT = os.path.dirname(
    path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
)


def get_model_path(model):
    return path.join(PROJECT_ROOT, "data/models", model)


# def get_flw_path(person=""):
#    return path.join(PROJECT_ROOT, "/data/datasets/flw/flw_mtcnnpy_160/", person)


def get_face_database(db):
    return path.join(PROJECT_ROOT, "/data/database/", db)


def get_flw_sample_path(flw_path, person, img=None):
    if not isinstance(person, list):
        person = [person]

    if img is None:
        img = [0] * len(person)
    elif not isinstance(img, list):
        img = [img]

    flw_dir = os.listdir(flw_path)
    person_path = [path.join(flw_path, flw_dir[i]) for i in person]

    person_img = [path.join(p, os.listdir(p)[i]) for p, i in zip(person_path, img)]

    return person_img
