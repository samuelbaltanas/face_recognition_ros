import os
import sys

import cv2
import mxnet as mx
import numpy as np
from mxnet import ndarray as nd
from sklearn import preprocessing

import tvm

# import vta
from face_recognition_ros.utils import config, files
from tvm import relay

# from tvm.contrib import graph_runtime

prefix = os.path.join(files.PROJECT_ROOT, "data", "models", "model-r100-ii", "model")
# prefix = os.path.join(files.PROJECT_ROOT, "data", "models", "model-y1-test2", "model")
epoch = 0
image_size = [112, 112]


class FaceEncoder:
    def __init__(self, conf=None):
        if conf is None:
            conf = config.CONFIG

        ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()

        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

        all_layers = sym.get_internals()
        sym = all_layers["fc1_output"]
        self.model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)

        self.model.bind(data_shapes=[("data", (1, 3, image_size[0], image_size[1]))])
        self.model.set_params(arg_params, aux_params)

    def predict(self, face_images, batch_size=1):
        embs = []

        for i, face in enumerate(face_images):
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = np.transpose(face, (2, 0, 1))
            face = np.expand_dims(face, 0)
            face_images[i] = face.astype("float32")

        dataloader = mx.gluon.data.DataLoader(
            np.vstack(face_images),
            batch_size=batch_size,
            last_batch="keep",
            shuffle=False,
        )

        for faces in dataloader:

            db = mx.io.DataBatch(data=(faces,))
            self.model.forward(db, is_train=False)
            embedding = self.model.get_outputs()[0].asnumpy()
            assert embedding.shape[1] == 512

            embedding = preprocessing.normalize(embedding, axis=1)
            embs.append(embedding)

        return np.vstack(embs)


def compileTVM():
    # prefix, epoch = "emore1",0
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    image_size = 112, 112
    opt_level = 3

    shape_dict = {"data": (1, 3, *image_size)}
    target = "cuda"
    # target = tvm.target.create("llvm -mcpu=haswell")
    # "target" means your target platform you want to compile.

    # target = tvm.target.create("llvm -mcpu=broadwell")
    nnvm_sym, nnvm_params = relay.frontend.from_mxnet(
        sym, shape_dict, arg_params=arg_params, aux_params=aux_params
    )
    with relay.build_config(opt_level=opt_level):
        graph, lib, params = relay.build(nnvm_sym, target, params=nnvm_params)
    lib.export_library("./deploy_lib.so")
    print("lib export succeefully")
    with open("./deploy_graph.json", "w") as fo:
        fo.write(graph)
    with open("./deploy_param.params", "wb") as fo:
        fo.write(relay.save_param_dict(params))


if __name__ == "__main__":
    compileTVM()
