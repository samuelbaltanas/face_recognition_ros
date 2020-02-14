from __future__ import print_function

import logging.config
import os
import sys

import yaml

from face_recognition_ros.utils import files

CONFIG_PATH = os.path.join(files.PROJECT_ROOT, "cfg")
DEFAULT_CFG = "test_cpu_fast.yaml"
LOGGING_CONFIG = os.path.join(files.PROJECT_ROOT, "cfg", "logging.yaml")
CONFIG = {}


def logger_config(conf_path=LOGGING_CONFIG):
    with open(conf_path) as conf:
        logging.config.dictConfig(yaml.safe_load(conf))


def load_config(cfg=DEFAULT_CFG):
    """Update configuration from a dictionary or a YAML file"""
    global CONFIG

    if type(cfg) is dict:
        CONFIG = _load_from_dict(cfg)
    elif type(cfg) is str:
        CONFIG = _load_from_file(cfg)
    else:
        raise TypeError(
            "A configuration can only be loaded from a path or a dictionary."
        )

    return CONFIG


def _load_from_dict(conf):
    # TODO: Assertions + load only useful fields
    return conf


def _load_from_file(path):
    if os.path.dirname(path) == "":
        FULL_PATH = os.path.join(CONFIG_PATH, path)
    else:
        FULL_PATH = path

    try:
        with open(FULL_PATH, "r") as f:
            conf = yaml.safe_load(f)
    except Exception:
        logging.error("Configuration file cannot be loaded.")
        sys.exit(-1)

    return conf
