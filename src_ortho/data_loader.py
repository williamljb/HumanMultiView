"""
Data loader with data augmentation.
Only used for training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
from glob import glob

import tensorflow as tf

from .tf_smpl.batch_lbs import batch_rodrigues_back, batch_rodrigues
from .util import data_utils
import numpy as np

_MV_DATASETS = ['mpi_inf_3dhp', 'synthetic']


def num_examples(datasets):
    _NUM_TRAIN = {
        'mpi_inf_3dhp': 15451, # 1618 for val
        'synthetic': 28972, #3952 #43856, # 5942 for val
        'h36m': 44423,
        'coco': 0,
    }

    if not isinstance(datasets, list):
        datasets = [datasets]
    total = 0

    use_dict = _NUM_TRAIN

    for d in datasets:
        total += use_dict[d]
    return total


class DataLoader(object):
    def __init__(self, config):
        self.config = config

        self.use_3d_label = config.use_3d_label

        self.dataset_dir = config.data_dir
        self.datasets = config.datasets
        self.mocap_datasets = config.mocap_datasets
        self.batch_size = config.batch_size
        self.data_format = config.data_format
        self.output_size = config.img_size
        # Jitter params:
        self.trans_max = config.trans_max
        self.scale_range = [config.scale_min, config.scale_max]

        self.image_normalizing_fn = data_utils.rescale_image

    def load(self):
        if self.use_3d_label:
            image_loader = self.get_loader_w3d()
        else:
            image_loader = self.get_loader()

        return image_loader

    def get_loader(self):
        """
        Outputs:
          image_batch: batched images as per data_format
          label_batch: batched keypoint labels N x K x 3
        """
        files = data_utils.get_all_files(self.dataset_dir, self.datasets)

        do_shuffle = True
        fqueue = tf.train.string_input_producer(
            files, shuffle=do_shuffle, name="input")
        image, label = self.read_data(fqueue, has_3d=False)
        min_after_dequeue = 5000
        num_threads = 8
        capacity = min_after_dequeue + 3 * self.batch_size

        pack_these = [image, label]
        pack_name = ['image', 'label']

        all_batched = tf.train.shuffle_batch(
            pack_these,
            batch_size=self.batch_size,
            num_threads=num_threads,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            enqueue_many=False,
            name='input_batch_train')
        batch_dict = {}
        for name, batch in zip(pack_name, all_batched):
            batch_dict[name] = batch

        return batch_dict

    def get_loader_w3d(self):
        """
        Similar to get_loader, but outputs are:
          image_batch: batched images as per data_format
          label_batch: batched keypoint labels N x K x 3
          label3d_batch: batched keypoint labels N x (216 + 10 + 42)
                         216=24*3*3 pose, 10 shape, 42=14*3 3D joints
                         (3D datasets only have 14 joints annotated)
          has_gt3d_batch: batched indicator for
                          existence of [3D joints, 3D SMPL] labels N x 2 - bool
                          Note 3D SMPL is only available for H3.6M.


        Problem is that those datasets without pose/shape do not have them
        in the tfrecords. There's no way to check for this in TF,
        so, instead make 2 string_input_producers, one for data without 3d
        and other for data with 3d.
        And send [2 x *] to train.*batch
        """
        datasets_nomv = [d for d in self.datasets if d not in _MV_DATASETS]
        datasets_yesmv = [d for d in self.datasets if d in _MV_DATASETS]

        # TODO: synthetic data has smpl but no 3d joint!
        files_nomv = data_utils.get_all_files(self.dataset_dir, datasets_nomv)
        files_yesmv = data_utils.get_all_files(self.dataset_dir,
                                               datasets_yesmv)

        # Make sure we have dataset with 3D.

        do_shuffle = True

        if len(files_yesmv) != 0:
            fqueue_yesmv = tf.train.string_input_producer(
                files_yesmv, shuffle=do_shuffle, name="input_wmv")
            image, label, label3d, has_smpl3d, pose,_ = self.read_data(
                fqueue_yesmv, has_multiview=True)
            if len(files_nomv) != 0:
                fqueue_nomv = tf.train.string_input_producer(
                    files_nomv, shuffle=do_shuffle, name="input_woutmv")
                image_nomv, label_nomv, label3d_nomv, has_smpl3d_nomv, pose_nomv, has_3djoint = self.read_data(
                    fqueue_nomv, has_multiview=False)
                image = tf.parallel_stack([image, image_nomv])
                label = tf.parallel_stack([label, label_nomv])
                label3d = tf.parallel_stack([label3d, label3d_nomv])
                has_smpl3d_nomv = tf.concat([has_3djoint, has_smpl3d_nomv], axis=0)
                has_3dgt = tf.parallel_stack([has_smpl3d, has_smpl3d_nomv])
                pose = tf.parallel_stack([pose, pose_nomv])
            else:
                assert False
                # If no "no3d" images, need to make them 1 x *
                image = tf.expand_dims(image, 0)
                label = tf.expand_dims(label, 0)
                label3d = tf.expand_dims(label3d, 0)
                has_3dgt = tf.expand_dims(has_smpl3d, 0)
                pose = tf.expand_dims(pose, 0)
        else:
            fqueue_nomv = tf.train.string_input_producer(
                files_nomv, shuffle=do_shuffle, name="input_woutmv")
            image, label, label3d, has_smpl3d_nomv, pose, has_3djoint = self.read_data(
                fqueue_nomv, has_multiview=False)
            has_3dgt = tf.concat([has_3djoint, has_smpl3d_nomv], axis=0)
            image = tf.expand_dims(image, 0)
            label = tf.expand_dims(label, 0)
            label3d = tf.expand_dims(label3d, 0)
            has_3dgt = tf.expand_dims(has_3dgt, 0)
            pose = tf.expand_dims(pose, 0)
        # Combine 3D bools.
        # each is 2 x 1, column is [3d_joints, 3d_smpl]

        min_after_dequeue = 2000
        capacity = min_after_dequeue + 3 * self.batch_size

        print('image.shape=',image.shape)
        print('label.shape=',label.shape)
        print('label3d.shape=',label3d.shape)
        print('has_3dgt.shape=',has_3dgt.shape)
        print('pose.shape=',pose.shape)
        image_batch, label_batch, label3d_batch, bool_batch, pose_batch = tf.train.shuffle_batch(
            [image, label, label3d, has_3dgt, pose],
            batch_size=self.batch_size,
            num_threads=8,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            enqueue_many=True,
            name='input_batch_train_3d')

        if self.data_format == 'NCHW':
            image_batch = tf.transpose(image_batch, [0, 3, 1, 2])
        elif self.data_format == 'NHWC':
            pass
        else:
            raise Exception("[!] Unkown data_format: {}".format(
                self.data_format))

        batch_dict = {
            'image': image_batch,
            'label': label_batch,
            'label3d': label3d_batch,
            'has3d': bool_batch,
            'oripose': pose_batch,
        }

        return batch_dict

    def get_smpl_loader(self):
        """
        Loads dataset in form of queue, loads shape/pose of smpl.
        returns a batch of pose & shape
        """

        return None
        data_dirs = [
            join(self.dataset_dir, 'mocap_neutrMosh',
                 'neutrSMPL_%s_*.tfrecord' % dataset)
            for dataset in self.mocap_datasets
        ]
        files = []
        for data_dir in data_dirs:
            files += glob(data_dir)

        if len(files) == 0:
            print('Couldnt find any files!!')
            import ipdb
            ipdb.set_trace()

        return self.get_smpl_loader_from_files(files)

    def get_smpl_loader_from_files(self, files):
        """
        files = list of tf records.
        """
        with tf.name_scope('input_smpl_loader'):
            filename_queue = tf.train.string_input_producer(
                files, shuffle=True)

            mosh_batch_size = self.batch_size * self.config.num_stage

            min_after_dequeue = 1000
            capacity = min_after_dequeue + 3 * mosh_batch_size

            pose, shape = data_utils.read_smpl_data(filename_queue)
            pose_batch, shape_batch = tf.train.batch(
                [pose, shape],
                batch_size=mosh_batch_size,
                num_threads=4,
                capacity=capacity,
                name='input_smpl_batch')

            return pose_batch, shape_batch

    def read_data(self, filename_queue, has_multiview=False):
        with tf.name_scope(None, 'read_data', [filename_queue]):
            reader = tf.TFRecordReader()
            _, example_serialized = reader.read(filename_queue)
            if has_multiview:
                (image, image_size, label, center, fname, pose, shape, gt3d, 
                    has_smpl3d, has_3djoint) = data_utils.parse_example_proto(example_serialized, num_view=4)

                # Need to send pose bc image can get flipped.
                image, label, pose, gt3d = self.image_preprocessing(
                    image, image_size, label, center, poses=pose, gt3ds=gt3d)
                # random roll
                ind = tf.constant([0,1,2,3], dtype=tf.int32)
                # st = tf.random_uniform([1], maxval=4, dtype=tf.int32)
                # ind = tf.manip.roll(ind, shift=st, axis=[0])
                p = tf.random_uniform([], dtype=tf.float32)
                ind = tf.cond(p < 0.3, lambda: tf.tile(ind[:1], [4]), lambda: ind)
                image = tf.gather(image, ind)
                image_size = tf.gather(image_size, ind)
                label = tf.gather(label, ind)
                center = tf.gather(center, ind)
                fname = tf.gather(fname, ind)
                pose = tf.gather(pose, ind)
                shape = tf.gather(shape, ind)
                gt3d = tf.gather(gt3d, ind)

            else:
                (image, image_size, label, center, fname, pose, shape, gt3d, 
                    has_smpl3d, has_3djoint) = data_utils.parse_example_proto(example_serialized, num_view=1)
                image = tf.tile(image, [4, 1, 1, 1])
                image_size = tf.tile(image_size, [4, 1])
                label = tf.tile(label, [4, 1, 1])
                center = tf.tile(center, [4, 1, 1])
                fname = tf.tile(fname, [4])
                # h36m upside down
                tmp = batch_rodrigues_back(tf.matmul(
                    batch_rodrigues(tf.constant([[np.pi,0,0]], tf.float32)),
                    batch_rodrigues(pose[:,:3])
                    )) # 1*3
                # h36m pose independent on cam, not useful
                has_smpl3d = [False]
                # with tf.control_dependencies([tf.assert_equal(tmp, pose[:,:3])]):
                pose = tf.concat([tmp, pose[:,3:]], axis=1)
                pose = tf.tile(pose, [4, 1])
                shape = tf.tile(shape, [4, 1])
                gt3d = tf.tile(gt3d, [4, 1, 1])
                image, label, pose, gt3d = self.image_preprocessing(
                    image, image_size, label, center, poses=pose, gt3ds=gt3d)

            # Convert pose to rotation.
            # Do not ignore the global!!
            rotations = batch_rodrigues(tf.reshape(pose, [-1, 3]))
            gt3d_flat = tf.reshape(gt3d, [4, -1])
            # Label 3d is:
            #   [rotations, shape-beta, 3Djoints]
            #   [216=24*3*3, 10, 42=14*3]
            label3d = tf.concat(
                [tf.reshape(rotations, [4, -1]), shape, gt3d_flat], 1)

            # label should be K x 3
            label = tf.transpose(label, perm=[0, 2, 1])

            return image, label, label3d, has_smpl3d, pose, has_3djoint

    def image_preprocessing(self,
                            images,
                            image_sizes,
                            labels,
                            centers,
                            poses=None,
                            gt3ds=None):
        num_cam = 4
        margin = tf.to_int32(self.output_size / 2)
        scale_factor = 1
        with tf.name_scope(None, 'image_preprocessing',
                           [images, image_sizes, labels, centers]):
            for i in range(num_cam):
                visibility = labels[i, 2, :]
                keypoints = labels[i, :2, :]

                # Randomly shift center.
                print('Using translation jitter: %d' % self.trans_max)
                center = data_utils.jitter_center(centers[i], self.trans_max)
                # randomly scale image.
                image, keypoints, center = data_utils.jitter_scale(
                    images[i], image_sizes[i], keypoints, center, self.scale_range, self.output_size, scale_factor)

                # Pad image with safe margin.
                # Extra 50 for safety.
                margin_safe = margin + self.trans_max + 200
                image_pad = data_utils.pad_image_edge(image, margin_safe)
                center_pad = center + margin_safe
                keypoints_pad = keypoints + tf.to_float(margin_safe)

                start_pt = center_pad - margin

                # Crop image pad.
                start_pt = tf.squeeze(start_pt)
                bbox_begin = tf.stack([start_pt[1], start_pt[0], 0])
                bbox_size = tf.stack([self.output_size, self.output_size, 3])

                crop = tf.slice(image_pad, bbox_begin, bbox_size)
                x_crop = keypoints_pad[0, :] - tf.to_float(start_pt[0])
                y_crop = keypoints_pad[1, :] - tf.to_float(start_pt[1])

                crop_kp = tf.stack([x_crop, y_crop, visibility])

                # Normalize kp output to [-1, 1]
                final_vis = tf.cast(crop_kp[2, :] > 0, tf.float32)
                final_label = tf.stack([
                    2.0 * (crop_kp[0, :] / self.output_size) - 1.0,
                    2.0 * (crop_kp[1, :] / self.output_size) - 1.0, final_vis
                ])
                # Preserving non_vis to be 0.
                final_label = final_vis * final_label

                # rescale image from [0, 1] to [-1, 1]
                crop = self.image_normalizing_fn(crop)
                if i == 0:
                    crops = tf.expand_dims(crop, 0)
                    final_labels = tf.expand_dims(final_label, 0)
                else:
                    crops = tf.concat([crops, tf.expand_dims(crop, 0)], 0)
                    final_labels = tf.concat([final_labels, tf.expand_dims(final_label, 0)], 0)
        if poses is not None:
            return crops, final_labels, poses, gt3ds
        else:
            return crops, final_labels
