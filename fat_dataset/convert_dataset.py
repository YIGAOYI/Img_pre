"""
A lot of the following code is a rewrite of:
https://github.com/deepmind/gqn-datasets/data_reader.py
"""

import numpy as np
import os
import collections
import torch
import gzip
from PIL import Image

import tensorflow as tf

from sys import version_info
if version_info.major == 3 and version_info.minor < 6:
    from fstring import fstring


DatasetInfo = collections.namedtuple(
    'DatasetInfo',
    ['basepath', 'train_size', 'test_size', 'frame_size', 'sequence_size']
)
Context = collections.namedtuple('Context', ['frames', 'cameras'])
Scene = collections.namedtuple('Scene', ['frames', 'cameras'])
Query = collections.namedtuple('Query', ['context', 'query_camera'])
TaskData = collections.namedtuple('TaskData', ['query', 'target'])


_DATASETS = dict(
    fat=DatasetInfo(
        basepath='fat'
    )
)

im_height = 256
im_width = 256


def _convert_frame_data(jpeg_data):
  decoded_frames = tf.image.decode_jpeg(jpeg_data)
  return tf.image.convert_image_dtype(decoded_frames, dtype=tf.float32)


def preprocess_frames(dataset_info, example, jpeg='False'):
    """Instantiates the ops used to preprocess the frames data."""
    frames = tf.concat(example['frames'], axis=0)
    if not jpeg:
        frames = tf.map_fn(_convert_frame_data, tf.reshape(frames, [-1]),dtype=tf.float32, back_prop=False)
        dataset_image_dimensions = tuple([dataset_info.frame_size] * 2 + [3])
        frames = tf.reshape(frames, (-1, dataset_info.sequence_size) + dataset_image_dimensions)
        if 64 and 64 != dataset_info.frame_size:
            frames = tf.reshape(frames, (-1,) + dataset_image_dimensions)
            new_frame_dimensions = (64,) * 2 + (3,)
            frames = tf.image.resize_bilinear(frames, new_frame_dimensions[:2], align_corners=True)
            frames = tf.reshape(frames, (-1, dataset_info.sequence_size) + new_frame_dimensions)
    return frames


def preprocess_cameras(dataset_info, example, raw):
    """Instantiates the ops used to preprocess the cameras data."""
    raw_pose_params = example['cameras']
    raw_pose_params = tf.reshape(
        raw_pose_params,
        [-1, dataset_info.sequence_size, 5])
    if not raw:
        pos = raw_pose_params[:, :, 0:3]
        yaw = raw_pose_params[:, :, 3:4]
        pitch = raw_pose_params[:, :, 4:5]
        cameras = tf.concat(
            [pos, tf.sin(yaw), tf.cos(yaw), tf.sin(pitch), tf.cos(pitch)], axis=2)
        return cameras
    else:
        return raw_pose_params


def _get_dataset_files(dt_info, mode, root):
    """
    Generates lists of files for a given dataset version.
    A symlink to the root of fat should be put into fat_dataset
    """
    base_path = dt_info.basepath
    base = os.path.join(root, base_path, mode)
    files = sorted(os.listdir(base))

    return [[fname, os.path.join(base, fname)] for fname in files]


def encapsulate(frames, cameras):
    return Scene(cameras=cameras, frames=frames)


def convert_raw_to_numpy(dataset_info, raw_data, path, jpeg=False):
    feature_map = {
        'frames': tf.FixedLenFeature(
            shape=dataset_info.sequence_size, dtype=tf.string),
        'cameras': tf.FixedLenFeature(
            shape=[dataset_info.sequence_size * 5],
            dtype=tf.float32)
    }
    example = tf.parse_single_example(raw_data, feature_map)
    frames = preprocess_frames(dataset_info, example, jpeg)
    cameras = preprocess_cameras(dataset_info, example, jpeg)
    with tf.train.SingularMonitoredSession() as sess:
        frames = sess.run(frames)
        cameras = sess.run(cameras)
    scene = encapsulate(frames, cameras)
    with gzip.open(path, 'wb') as f:
        torch.save(scene, f)


def show_frame(frames, scene, views):
    import matplotlib.pyplot as plt
    plt.imshow(frames[scene, views])
    plt.show()


def show_img(img):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()


def adjust_depth_img(img):
    img = img.resize((im_height, im_width))
    img_np = np.array(img.getdata()).reshape(
        (im_height, im_width, 1)).astype(np.float32)
    return img_np


def adjust_rgb_img(img):
    img = img.resize((im_height, im_width))
    img_np = np.array(img.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
    return img_np.astype(np.float32) / 255.


def load_vp(file_path):
    import json
    f = open(file_path, encoding='utf-8')
    jf = json.load(f)
    camera_data = jf['camera_data']
    location_worldframe = camera_data["location_worldframe"]
    quaternion_xyzw_worldframe = camera_data["quaternion_xyzw_worldframe"]
    return location_worldframe + quaternion_xyzw_worldframe


if __name__ == '__main__':
    import sys

    """
    Use this script with 2 params: the root path to the dataset, subset name
    """
    if len(sys.argv) < 3:
        print(' [!] you need to give a dataset')
        exit()

    root_dir = sys.argv[1]
    DATASET = sys.argv[2]
    dataset_info = _DATASETS[DATASET]

    torch_dataset_path = fstring("{DATASET}-torch")
    torch_dataset_path_train = fstring('{torch_dataset_path}/train')
    torch_dataset_path_test = fstring('{torch_dataset_path}/test')

    os.mkdir(os.path.join(root_dir, torch_dataset_path))
    os.mkdir(os.path.join(root_dir, torch_dataset_path_train))
    os.mkdir(os.path.join(root_dir, torch_dataset_path_test))

    # train
    file_list = _get_dataset_files(dataset_info, 'train', '.')

    tot = 0
    count = 0
    depth_data = []
    rgb_data = []
    view_point_data = []

    for file in file_list:
        fn = file[0]  # name of the file
        fp = file[1]  # absolute path of the file
        # Omit seg image files by its name
        if 'seg' in fn:
            continue
        # Else, phase depth and rgb images and the json file
        if fn.find('depth') >= 0:
            image = Image.open(fp)
            # show_img(image)
            im = adjust_depth_img(image)
            depth_data.append(im)
        elif 'jpg' in fn:
            image = Image.open(fp)
            im = adjust_rgb_img(image)
            rgb_data.append(im)
        else:
            vp = load_vp(fp)
            view_point_data.append(vp)

            count += 1
            if count == 200:
                np.save(torch_dataset_path_train + '/depth_' + str(tot), depth_data)
                np.save(torch_dataset_path_train + '/rgb_' + str(tot), rgb_data)
                np.save(torch_dataset_path_train + '/vp_' + str(tot), view_point_data)
                count = 0
                tot += 1

    print(fstring(' [-] {tot*200} scenes in the train dataset'))

    # # test
    file_list = _get_dataset_files(dataset_info, 'test', '.')

    tot = 0
    count = 0
    depth_data = []
    rgb_data = []
    view_point_data = []

    for file in file_list:
        fn = file[0]  # name of the file
        fp = file[1]  # absolute path of the file
        # Omit seg image files by its name
        if 'seg' in fn:
            continue
        # Else, phase depth and rgb images and the json file
        if 'depth' in fn:
            image = Image.open(fp)
            im = adjust_depth_img(image)
            depth_data.append(im)
        elif 'jpg' in fn:
            image = Image.open(fp)
            im = adjust_rgb_img(image)
            rgb_data.append(im)
        else:
            vp = load_vp(fp)
            view_point_data.append(vp)

            count += 1
            if count == 200:
                np.save(torch_dataset_path_test + '/depth_' + str(tot), depth_data)
                np.save(torch_dataset_path_test + '/rgb_' + str(tot), rgb_data)
                np.save(torch_dataset_path_test + '/vp_' + str(tot), view_point_data)
                count = 0
                tot += 1

    print(fstring(' [-] {tot*200} scenes in the test dataset'))
