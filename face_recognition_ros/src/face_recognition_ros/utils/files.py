# TODO: Better checks + incorporate config files
import os
from os import path

PROJECT_ROOT = os.path.dirname(
    path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
)


def get_model_path(model_path, model):
    if not model_path:
        return path.join(PROJECT_ROOT, "data/models", model)
    else:
        return path.join(model_path, model)


def get_face_database(db):
    return path.join(PROJECT_ROOT, "data/database", db)


def image_folder_traversal(image_dir):
    for dir_name, _, file_list in os.walk(image_dir):
        if dir_name == image_dir:
            continue
        label = os.path.basename(dir_name)
        files = (os.path.join(dir_name, file_name) for file_name in file_list)
        yield label, files
