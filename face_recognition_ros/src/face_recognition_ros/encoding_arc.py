import os

import cv2
import numpy as np
import mxnet as mx
from sklearn import preprocessing

# import nnvm.compiler
# import nnvm.testing
import tvm
from tvm import relay
from tvm.contrib import graph_runtime
from mxnet import ndarray as nd
import vta

from face_recognition_ros.utils import files, config

prefix = os.path.join(
    files.PROJECT_ROOT, "data", "models", "model-r100-ii", "model"
)
# prefix = os.path.join(files.PROJECT_ROOT, "data", "models", "model-y1-test2", "model")
epoch = 0
image_size = [112, 112]


class EncodingArc:
    def __init__(self, conf=None):
        if conf is None:
            conf = config.CONFIG

        # ctx = mx.cpu()
        ctx = mx.gpu()

        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

        all_layers = sym.get_internals()
        sym = all_layers["fc1_output"]
        self.model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)

        self.model.bind(
            data_shapes=[("data", (1, 3, image_size[0], image_size[1]))]
        )
        self.model.set_params(arg_params, aux_params)

    def predict(self, face_images):

        for i, face in enumerate(face_images):
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = np.transpose(face, (2, 0, 1))
            face = np.expand_dims(face, 0)
            face_images[i] = face

        input_blob = np.vstack(face_images)

        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()

        embedding = preprocessing.normalize(embedding, axis=1)

        return embedding


def compileTVM():
    # prefix, epoch = "emore1",0
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    image_size = (112, 112)
    opt_level = 3

    shape_dict = {'data': (1, 3, *image_size)}
    target = "cuda"
    # target = tvm.target.create("llvm -mcpu=haswell")
    # "target" means your target platform you want to compile.

    # target = tvm.target.create("llvm -mcpu=broadwell")
    nnvm_sym, nnvm_params = relay.frontend.from_mxnet(sym, arg_params, aux_params)
    with relay.build_config(opt_level=opt_level):
        graph, lib, params = relay.build(nnvm_sym, target, shape_dict, params=nnvm_params)
    lib.export_library("./deploy_lib.so")
    print('lib export succeefully')
    with open("./deploy_graph.json", "w") as fo:
        fo.write(graph.json())
    with open("./deploy_param.params", "wb") as fo:
        fo.write(relay.save_param_dict(params))


if __name__ == "__main__":
    compileTVM()
