import os
from os import path


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_model_path(model):
    return '{}/data/models/{}'.format(PROJECT_ROOT, model)


def get_flw_path(person=''):
    return ('{}/data/datasets/flw/flw_mtcnnpy_160/{}'
            .format(PROJECT_ROOT, person))


def get_flw_sample_path(person, img=None):
    if not isinstance(person, list):
        person = [person]

    if img is None:
        img = [0]*len(person)
    elif not isinstance(img, list):
        img = [img]

    flw_path = get_flw_path()

    flw_dir = os.listdir(flw_path)
    person_path = [path.join(flw_path, flw_dir[i]) for i in person]

    person_img = [path.join(p, os.listdir(p)[i]) for p, i in zip(person_path, img)]

    return person_img


def get_face_database(db):
    return '{}/data/database/{}'.format(PROJECT_ROOT, db)
