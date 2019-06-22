import numpy as np


def prewhiten(img):
    mean = np.mean(img)
    std = np.std(img)
    std_adj = np.maximum(std, 1.0 / np.sqrt(img.size))
    y = np.multiply(np.subtract(img, mean), 1 / std_adj)
    return y


def preprocess_face(image, fixed_standardization=True):
    # Necessary for model 20180402-114759 as explained in:
    # https://github.com/davidsandberg/facenet/wiki/Training-using-the-VGGFace2-dataset
    if fixed_standardization:
        image = (np.float32(image) - 127.5) / 128.0
    image = prewhiten(image)
    return image
