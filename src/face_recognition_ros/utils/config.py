from __future__ import print_function

import sys
import os

import yaml

from face_recognition_ros.utils import files

CONFIG_PATH = files.PROJECT_ROOT + "/cfg/"
DEFAULT_CFG = "test_cpu_fast.yaml"

def cat_config(cfg=DEFAULT_CFG):
    with open(CONFIG_PATH + cfg) as f:
        print(f.read())

def list_config():
    print(os.listdir(CONFIG_PATH))

def load_config(cfg=DEFAULT_CFG):
    global CONFIG
    try:
        with open(CONFIG_PATH + cfg, "r") as f:
            CONFIG = yaml.load(f)
    except Exception as e:
        print("[ERR] Configuration file cannot be loaded.", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(-1)

    return CONFIG
