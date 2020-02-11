from typing import List
from urllib import request, parse
import os
from os import path
import zipfile
import tempfile

import yaml

from face_recognition_ros.utils import files

__MODEL_DIR = path.join(files.PROJECT_ROOT, "data", "models")
__MODEL_LIST = path.join(files.PROJECT_ROOT, "data", "model_list.yaml")


def list_models(installed=False) -> List[str]:
    with open(__MODEL_LIST) as list_file:
        models = yaml.load(list_file)

    if installed:
        pass

    return models


def get_model():
    pass


def request_loader(dst_folder: str, url: str) -> None:
    response = request.urlopen(url)

    _, _, fpath, _, _ = parse.urlsplit(url)
    filename = path.basename(fpath)

    if filename.split(".")[-1] == "zip":
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp_name = temp.name
            temp.write(response.read())
            temp.flush()

        os.mkdir(dst_folder)

        with zipfile.ZipFile(temp_name, 'r') as zip_ref:
            zip_ref.extractall(dst_folder)

        os.remove(temp_name)
    else:
        os.mkdir(dst_folder)
        with open(path.join(dst_folder, filename), "wb+") as f:
            f.write(response.read())
            f.flush()
