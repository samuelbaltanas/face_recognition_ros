import os
import tempfile
import zipfile
from os import path
from typing import List
from urllib import parse, request

import yaml

from face_recognition_ros.utils import files

__MODEL_DIR = path.join(files.PROJECT_ROOT, "data", "models")
__MODEL_LIST = path.join(files.PROJECT_ROOT, "data", "model_list.yaml")

__MODELS = None


def _load_list() -> dict:
    global __MODELS
    if __MODELS is None:
        with open(__MODEL_LIST) as list_file:
            __MODELS = yaml.load(list_file)  # type: dict

    return __MODELS


def list_models(installed=False) -> List[str]:
    models = _load_list()

    if installed:
        droplist = [
            i
            for i in models.keys()
            if not path.exists(path.join(__MODEL_DIR, i))
        ]
        for i in droplist:
            models.pop(i)

    return models


def get_model(model: str):
    models = _load_list()

    if model not in models:
        print("Model {} not found in model list.".format(model))

    mod_path = path.join(__MODEL_DIR, model)

    if not path.exists(mod_path):
        request_loader(__MODEL_DIR, models[model]["url"])

    assert path.exists(mod_path)

    return mod_path


def request_loader(dst_folder: str, url: str) -> None:
    response = request.urlopen(url)

    _, _, fpath, _, _ = parse.urlsplit(url)
    filename = path.basename(fpath)

    if filename.split(".")[-1] == "zip":
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp_name = temp.name
            temp.write(response.read())
            temp.flush()

        # os.mkdir(dst_folder)

        with zipfile.ZipFile(temp_name, "r") as zip_ref:
            zip_ref.extractall(dst_folder)

        os.remove(temp_name)
    else:
        # os.mkdir(dst_folder)
        with open(path.join(dst_folder, filename), "wb+") as f:
            f.write(response.read())
            f.flush()
