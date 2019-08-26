"""
Helpers for tfrecord conversion.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities.
    Taken from
    https://github.com/tensorflow/models/blob/master/inception/inception/data/build_image_data.py
    """

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(
            image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._decode_jpeg_data, channels=3)

        self._encode_jpeg_data = tf.placeholder(dtype=tf.uint8)
        self._encode_jpeg = tf.image.encode_jpeg(
            self._encode_jpeg_data, format='rgb')

        self._decode_png_data = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(
            self._decode_png_data, channels=3)

        self._encode_png_data = tf.placeholder(dtype=tf.uint8)
        self._encode_png = tf.image.encode_png(self._encode_png_data)

    def png_to_jpeg(self, image_data):
        return self._sess.run(
            self._png_to_jpeg, feed_dict={
                self._png_data: image_data
            })

    def decode_jpeg(self, image_data):
        image = self._sess.run(
            self._decode_jpeg, feed_dict={
                self._decode_jpeg_data: image_data
            })
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def encode_jpeg(self, image):
        image_data = self._sess.run(
            self._encode_jpeg, feed_dict={
                self._encode_jpeg_data: image
            })
        return image_data

    def encode_png(self, image):
        image_data = self._sess.run(
            self._encode_png, feed_dict={
                self._encode_png_data: image
            })
        return image_data

    def decode_png(self, image_data):
        image = self._sess.run(
            self._decode_png, feed_dict={
                self._decode_png_data: image_data
            })
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def convert_to_example(image_data, image_path, height, width, label, center):
    """Build an Example proto for an image example.
    Args:
      image_data: string, JPEG encoding of RGB image;
      image_path: string, path to this image file
      labels: 3 x 14 joint location + visibility --> This could be 3 x 19
      height, width: integers, image shapes in pixels.
      center: 2 x 1 center of the tight bbox
    Returns:
      Example proto
    """
    from os.path import basename

    image_format = 'JPEG'
    add_face = False
    if label.shape[1] == 19:
        add_face = True
        # Split and save facepts on it's own.
        face_pts = label[:, 14:]
        label = label[:, :14]
    has_3d=np.zeros(1)
    feat_dict = {
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/center': int64_feature(center.astype(np.int)),
        'image/x': float_feature(label[0, :].astype(np.float)),
        'image/y': float_feature(label[1, :].astype(np.float)),
        'image/visibility': int64_feature(label[2, :].astype(np.int)),
        'image/format': bytes_feature([tf.compat.as_bytes(image_format)]),
        'image/filename': bytes_feature(
            [tf.compat.as_bytes(basename(image_path))]),
        'image/encoded': bytes_feature([tf.compat.as_bytes(image_data)]),
        'meta/has_3d': int64_feature(has_3d.astype(np.int)),
        'meta/has_3djoint': int64_feature(0),
    }
    if add_face:
        # 3 x 5
        feat_dict.update({
            'image/face_pts':
            float_feature(face_pts.ravel().astype(np.float))
        })

    example = tf.train.Example(features=tf.train.Features(feature=feat_dict))

    return example


def convert_to_example_wmosh(image_data, image_path, height, width, label,
                             center, gt3d, pose, shape, scale_factors,
                             start_pt, cam):
    """Build an Example proto for an image example.
    Args:
      image_data: 4*string, JPEG encoding of RGB image;
      image_path: 4*string, path to this image file
      labels: 4*3 x 14 joint location + visibility
      height, width: 4*integers, image shapes in pixels.
      center:4*2 x 1 center of the tight bbox
      gt3d: 4*14x3 3D joint locations
      scale_factors: 4*2 x 1, scale factor used to scale image.
      start_pt: 4*the left corner used to crop the _scaled_ image to 300x300
      cam: (4,3,), [f, px, py] intrinsic camera parameters.
    Returns:
      Example proto
    """
    from os.path import basename
    image_format = 'JPEG'
    label = np.array(label)
    if label.shape[1] != 3:
        label = label.T
    if label.shape[2] > 14:
        print('This shouldnt be happening')
        import ipdb
        ipdb.set_trace()
    num_cam = label.shape[0]
    # has_3d = [has_joint, has_smpl]
    has_3d = np.ones(2)
    if pose[0] is None:
        has_3d[1] = 0
        # Use -1 to save.
        pose = -np.ones((num_cam, 72))
        shape = -np.ones((num_cam, 10))
    if gt3d[0][0][0] == -1:
        gt3d = -np.ones((num_cam, 14, 3))
        has_3d[0] = 0

    example = tf.train.Example(
        features=tf.train.Features(feature={
            'image/height':
            int64_feature(height),
            'image/width':
            int64_feature(width),
            'image/center':
            int64_feature(np.array(center).ravel().astype(np.int)),
            'image/x':
            float_feature(label[:, 0, :].ravel().astype(np.float)),
            'image/y':
            float_feature(label[:, 1, :].ravel().astype(np.float)),
            'image/visibility':
            int64_feature(label[:, 2, :].ravel().astype(np.int)),
            'image/format':
            bytes_feature([tf.compat.as_bytes(image_format) for _ in image_path]),
            'image/filename':
            bytes_feature([tf.compat.as_bytes(basename(path)) for path in image_path]),
            'image/encoded':
            bytes_feature([tf.compat.as_bytes(data) for data in image_data]),
            'mosh/pose':
            float_feature(np.array(pose).ravel().astype(np.float)),
            'mosh/shape':
            float_feature(np.array(shape).ravel().astype(np.float)),
            'mosh/gt3d':
            float_feature(gt3d.ravel().astype(np.float)),
            'meta/scale_factors':
            float_feature(np.array(scale_factors).ravel().astype(np.float)),
            'meta/crop_pt':
            int64_feature(np.array(start_pt).ravel().astype(np.int)),
            'meta/has_3d':
            int64_feature(has_3d.astype(np.int)),
            'image/cam':
            float_feature(np.array(cam).ravel().astype(np.float)),
        }))

    return example


def resize_img(img, scale_factor):
    import cv2
    import numpy as np
    new_size = (np.floor(np.array(img.shape[0:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [
        new_size[0] / float(img.shape[0]), new_size[1] / float(img.shape[1])
    ]
    return new_img, actual_factor


def read_images_from_tfrecords(tf_path, num_views, img_size=224, sess=None):
    """
    Returns image, kp, and gt3d from the tf_paths

    This returns a preprocessed image, cropped around img_size.
    """
    from time import time
    from os.path import exists
    if not exists(tf_path):
        print('%s doesnt exist!' % tf_path)
        exit(1)

    if sess is None:
        sess = tf.Session()

    t0 = time()
    all_images, all_kps, all_gt3ds = [], [], []
    all_poses, all_shapes = [], []

    itr = 0

    # Decode op graph
    image_data_pl = tf.placeholder(dtype=tf.string)
    decode_op = tf.image.decode_jpeg(image_data_pl)

    for serialized_ex in tf.python_io.tf_record_iterator(tf_path):
        example = tf.train.Example()
        example.ParseFromString(serialized_ex)
        crops, kp_finals, gt3ds = [], [], []
        poses, shapes = [], []
        for i in range(num_views):
            image_data = example.features.feature['image/encoded'].bytes_list.value[i]
            image = sess.run(decode_op, feed_dict={image_data_pl:  image_data})

            x = example.features.feature['image/x'].float_list.value[14*i:14*(i+1)]
            y = example.features.feature['image/y'].float_list.value[14*i:14*(i+1)]
            vis = example.features.feature['image/visibility'].int64_list.value[14*i:14*(i+1)]
            center = example.features.feature['image/center'].int64_list.value[2*i:2*(i+1)]

            x = np.array(x)
            y = np.array(y)
            vis = np.array(vis, dtype='bool')
            center = np.array(center)

            # Crop img_size.
            # Pad in case.
            margin = int(img_size/2)
            image_pad = np.pad(image, ((200,200), (200,200), (0,0)), mode='edge')

            # figure out starting point
            start_pt = center - margin + 200
            end_pt = start_pt + 2*margin

            x_crop = x - start_pt[0] + 200
            y_crop = y - start_pt[1] + 200
            kp_crop = np.vstack([x_crop, y_crop])
            kp_final = 2 * (kp_crop / img_size) - 1
            kp_final = np.vstack((vis * kp_final, vis)).T
            # crop:
            crop = image_pad[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
            if crop.shape != (224,224,3):
                print("shape=",crop.shape)
                crop = np.zeros((224, 224, 3), np.float32)
            
            # Normalize image to [-1, 1]
            crop = 2 * ((crop / 255.) - 0.5)

            # Note: This says mosh but gt3d is the gt H3.6M joints & not from mosh.
            gt3d = example.features.feature['mosh/gt3d'].float_list.value[14*3*i:14*3*(i+1)]
            gt3d = np.array(gt3d).reshape(14, 3)

            pose = example.features.feature['mosh/pose'].float_list.value[72*i:72*(i+1)]
            shape = example.features.feature['mosh/shape'].float_list.value[10*i:10*(i+1)]

            poses.append(pose)
            shapes.append(shape)
            crops.append(crop)
            kp_finals.append(kp_final)
            gt3ds.append(gt3d)

        all_poses.append(poses)
        all_shapes.append(shapes)
        all_images.append(crops)
        all_kps.append(kp_finals)
        all_gt3ds.append(gt3ds)

        itr += 1

    images = np.array(all_images)
    print(images.shape)
    kps = np.array(all_kps)
    gt3ds = np.array(all_gt3ds)
    # B * view * size

    print('Read %d images, %g secs' % (images.shape[0], time()-t0))

    return images, kps, gt3ds, np.array(all_poses), np.array(all_shapes)
