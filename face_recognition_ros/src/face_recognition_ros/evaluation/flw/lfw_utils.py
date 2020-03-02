"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import, division, print_function

import os

import numpy as np
from scipy import interpolate
from sklearn import model_selection

from face_recognition_ros.third_party import facenet
from face_recognition_ros.utils.math import dist as distance


def evaluate(
    embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False,
):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = facenet.calculate_roc(
        thresholds,
        embeddings1,
        embeddings2,
        np.asarray(actual_issame),
        nrof_folds=nrof_folds,
        distance_metric=distance_metric,
        subtract_mean=subtract_mean,
    )
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = facenet.calculate_val(
        thresholds,
        embeddings1,
        embeddings2,
        np.asarray(actual_issame),
        1e-3,
        nrof_folds=nrof_folds,
        distance_metric=distance_metric,
        subtract_mean=subtract_mean,
    )
    return tpr, fpr, accuracy, val, val_std, far


def get_paths(lfw_dir, pairs_filename):
    nrof_skipped_pairs = 0

    for pair in read_pairs(pairs_filename):
        if len(pair) == 3:
            issame = True
            path0 = join_image_path(lfw_dir, pair[0], pair[1])
            path1 = join_image_path(lfw_dir, pair[0], pair[2])
        elif len(pair) == 4:
            issame = False
            path0 = join_image_path(lfw_dir, pair[0], pair[1])
            path0 = join_image_path(lfw_dir, pair[2], pair[3])

        if os.path.exists(path0) and os.path.exists(
            path1
        ):  # Only add the pair if both paths exist
            yield (path0, path1), issame
        else:
            nrof_skipped_pairs += 1

    if nrof_skipped_pairs > 0:
        print("Skipped %d image pairs" % nrof_skipped_pairs)


def read_pairs(pairs_filename):
    with open(pairs_filename, "r") as f:
        for i, line in enumerate(f.readlines()[1:]):
            if i % 100 == 0:
                print("File: {}".format(i))
            pair = line.strip().split()
            yield pair


def join_image_path(root, name, num):
    path = os.path.join(root, name, name + "_" + "%04d" % int(num))

    if os.path.exists(path + ".jpg"):
        return path + ".jpg"
    elif os.path.exists(path + ".png"):
        return path + ".png"
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)


def calculate_roc(
    thresholds,
    embeddings1,
    embeddings2,
    actual_issame,
    nrof_folds=10,
    distance_metric=0,
    subtract_mean=False,
):
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = model_selection.KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(
                np.concatenate([embeddings1[train_set], embeddings2[train_set]]),
                axis=0,
            )
        else:
            mean = 0.0
        dist = distance(embeddings1 - mean, embeddings2 - mean, distance_metric)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set]
            )
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            (
                tprs[fold_idx, threshold_idx],
                fprs[fold_idx, threshold_idx],
                _,
            ) = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set], actual_issame[test_set],
        )

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame))
    )
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(
    thresholds,
    embeddings1,
    embeddings2,
    actual_issame,
    far_target,
    nrof_folds=10,
    distance_metric=0,
    subtract_mean=False,
):
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = model_selection.KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(
                np.concatenate([embeddings1[train_set], embeddings2[train_set]]),
                axis=0,
            )
        else:
            mean = 0.0
        dist = distance(embeddings1 - mean, embeddings2 - mean, distance_metric)

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set]
            )
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind="slinear")
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set]
        )

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far
