from __future__ import print_function

import sys
import yaml

from utils import files

try:
    with open(files.PROJECT_ROOT + "cfg/test_cpu.yaml", "r") as f:
        CONFIG = yaml.load(f)
except Exception as e:
    print("[ERR] Configuration file cannot be loaded.", file=sys.stderr)
    print(e, file=sys.stderr)
    sys.exit(-1)
