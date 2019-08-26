""" Convert MPI_INF_3DHP to TFRecords """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, exists
from os import makedirs

import numpy as np

import tensorflow as tf

from .common import convert_to_example_wmosh, ImageCoder, resize_img
from .mpi_inf_3dhp.read_mpi_inf_3dhp import get_paths, read_mat, mpi_inf_3dhp_to_lsp_idx, read_camera

tf.app.flags.DEFINE_string('data_directory', '/scratch1/storage/mpi_inf_3dhp/',
                           'data directory: top of mpi-inf-3dhp')
tf.app.flags.DEFINE_string('output_directory',
                           '/scratch1/projects/tf_datasets/mpi_inf_3dhp/',
                           'Output data directory')

tf.app.flags.DEFINE_string('split', 'train', 'train or val')
tf.app.flags.DEFINE_integer('train_shards', 500,
                            'Number of shards in training TFRecord files.')

FLAGS = tf.app.flags.FLAGS
MIN_VIS_PTS = 8  # This many points must be within the image.

# To go to h36m joints:
# training joints have 28 joints
# test joints are 17 (H3.6M subset in CPM order)
joint_idx2lsp, test_idx2lsp = mpi_inf_3dhp_to_lsp_idx()


def sample_frames(gt3ds):
    use_these = np.zeros(gt3ds.shape[0], bool)
    # Always use_these first frame.
    use_these[0] = True
    prev_kp3d = gt3ds[0]
    for itr, kp3d in enumerate(gt3ds):
        if itr > 0:
            # Check if any joint moved more than 200mm.
            if not np.any(np.linalg.norm(prev_kp3d - kp3d, axis=1) >= 200):
                continue
        use_these[itr] = True
        prev_kp3d = kp3d

    return use_these


def get_all_data(base_dir, sub_id, seq_id, cam_ids, all_cam_info):
    img_dir, anno_path = get_paths(base_dir, sub_id, seq_id)
    # Get data for all cameras.
    frames, _, annot2, annot3 = read_mat(anno_path)

    all_gt2ds, all_gt3ds, all_img_paths = [], [], []
    all_cams = []
    for cam_id in cam_ids:
        base_path = join(img_dir, 'video_%d' % cam_id, 'frame_%06d.jpg')
        num_frames = annot2[cam_id].shape[0]
        gt2ds = annot2[cam_id].reshape(num_frames, -1, 2)
        gt3ds = annot3[cam_id].reshape(num_frames, -1, 3)
        # Convert N x 28 x . to N x 14 x 2, N x 14 x 3
        gt2ds = gt2ds[:, joint_idx2lsp, :]
        gt3ds = gt3ds[:, joint_idx2lsp, :]
        img_paths = [base_path % (frame + 1) for frame in frames]
        if gt3ds.shape[0] != len(img_paths):
            print('Not same paths?')
            import ipdb
            ipdb.set_trace()
        use_these = sample_frames(gt3ds)
        all_gt2ds.append(gt2ds[use_these])
        all_gt3ds.append(gt3ds[use_these])
        K = all_cam_info[cam_id]
        flength = 0.5 * (K[0, 0] + K[1, 1])
        ppt = K[:2, 2]
        flengths = np.tile(flength, (np.sum(use_these), 1))
        ppts = np.tile(ppt, (np.sum(use_these), 1))
        cams = np.hstack((flengths, ppts))
        all_cams.append(cams)
        all_img_paths.append(np.array(img_paths)[use_these].tolist())

    all_gt2ds = np.transpose(np.array(all_gt2ds), [1,0,2,3])
    all_gt3ds = np.transpose(np.array(all_gt3ds), [1,0,2,3])
    all_cams = np.transpose(np.array(all_cams), [1,0,2])
    all_img_paths = np.transpose(np.array(all_img_paths), [1,0])
    print(all_img_paths.shape)

    return all_img_paths, all_gt2ds, all_gt3ds, all_cams


def check_good(image, gt2d):
    h, w, _ = image.shape
    x_in = np.logical_and(gt2d[:, 0] < w, gt2d[:, 0] >= 0)
    y_in = np.logical_and(gt2d[:, 1] < h, gt2d[:, 1] >= 0)

    ok_pts = np.logical_and(x_in, y_in)

    return np.sum(ok_pts) >= MIN_VIS_PTS


def add_to_tfrecord(im_paths,
                    gt2ds,
                    gt3ds,
                    cams,
                    coder,
                    writer,
                    model=None,
                    sub_path=None):
    """
    gt2ds is 4 * 14 x 2 (lsp order)
    gt3ds is 4 * 14 x 3
    cam is (4,3,)
    returns:
      success = 1 if this is a good image
      0 if most of the kps are outside the image
    """
    # Read image
    images, labels, heights, widths = [], [], [], []
    center_scaleds, scale_factorss = [], []
    start_pts, cam_scaleds = [], []
    for path, gt2d, cam in zip(im_paths, gt2ds, cams):
        if not exists(path):
            print('!!--%s doesnt exist! Skipping..--!!' % path)
            return False
        with tf.gfile.FastGFile(path, 'rb') as f:
            image_data = f.read()
        image = coder.decode_jpeg(coder.png_to_jpeg(image_data))
        assert image.shape[2] == 3
        if np.mean(image) > 180:
            print("skipping {}".format(path))
            return False

        good = check_good(image, gt2d)
        if not good:
            if FLAGS.split == 'test':
                print('Why no good?? shouldnt happen')
                import ipdb
                ipdb.set_trace()
            print("gt out of range!")
            return False

        # All kps are visible in mpi_inf_3dhp.
        min_pt = np.min(gt2d, axis=0)
        max_pt = np.max(gt2d, axis=0)
        person_height = np.linalg.norm(max_pt - min_pt)
        center = (min_pt + max_pt) / 2.
        scale = 150. / person_height

        image_scaled, scale_factors = resize_img(image, scale)
        height, width = image_scaled.shape[:2]
        joints_scaled = np.copy(gt2d)
        joints_scaled[:, 0] *= scale_factors[0]
        joints_scaled[:, 1] *= scale_factors[1]
        center_scaled = np.round(center * scale_factors).astype(np.int)
        # scale camera: Flength, px, py
        cam_scaled = np.copy(cam)
        cam_scaled[0] *= scale
        cam_scaled[1] *= scale_factors[0]
        cam_scaled[2] *= scale_factors[1]

        # Crop 300x300 around the center
        margin = 150
        start_pt = np.maximum(center_scaled - margin, 0).astype(int)
        end_pt = (center_scaled + margin).astype(int)
        end_pt[0] = min(end_pt[0], width)
        end_pt[1] = min(end_pt[1], height)
        image_scaled = image_scaled[start_pt[1]:end_pt[1], start_pt[0]:end_pt[
            0], :]
        # Update others too.
        joints_scaled[:, 0] -= start_pt[0]
        joints_scaled[:, 1] -= start_pt[1]
        center_scaled -= start_pt
        # Update principal point:
        cam_scaled[1] -= start_pt[0]
        cam_scaled[2] -= start_pt[1]
        height, width = image_scaled.shape[:2]

        # Encode image:
        image_data_scaled = coder.encode_jpeg(image_scaled)
        label = np.vstack([joints_scaled.T, np.ones((1, joints_scaled.shape[0]))])
        images.append(image_data_scaled)
        labels.append(label)
        heights.append(height)
        widths.append(width)
        center_scaleds.append(center_scaled)
        scale_factorss.append(scale_factors)
        start_pts.append(start_pt)
        cam_scaleds.append(cam_scaled)
    # pose and shape is not existent.
    pose, shape = [None,None,None,None], [None,None,None,None]
    example = convert_to_example_wmosh(
        images, im_paths, heights, widths, labels, center_scaleds, gt3ds / 1000.,
        pose, shape, scale_factorss, start_pts, cam_scaleds)
    writer.write(example.SerializeToString())

    return True


def save_to_tfrecord(out_name, im_paths, gt2ds, gt3ds, cams, num_shards):
    coder = ImageCoder()
    i = 0
    # Count on shards
    fidx = 0
    # Count failures
    num_bad = 0
    while i < len(im_paths):
        tf_filename = out_name % fidx
        print('Starting tfrecord file %s' % tf_filename)
        with tf.python_io.TFRecordWriter(tf_filename) as writer:
            j = 0
            while i < len(im_paths) and j < num_shards:
                if i % 100 == 0:
                    print('Reading img %d/%d' % (i, len(im_paths)))
                success = add_to_tfrecord(im_paths[i], gt2ds[i], gt3ds[i],
                                          cams[i], coder, writer)
                i += 1
                if success:
                    j += 1
                else:
                    num_bad += 1

        fidx += 1

    print('Done, wrote to %s, num skipped %d' % (out_name, num_bad))


def process_mpi_inf_3dhp_train(data_dir, out_dir, is_train=False):
    if is_train:
        out_dir = join(out_dir, 'train')
        print('!train set!')
        sub_ids = range(1, 8)  # No S8!
        seq_ids = range(1, 3)
        cam_ids = [8,0,2,7]
    else:  # Full set!!
        out_dir = join(out_dir, 'val')
        print('doing the full train-val set!')
        sub_ids = range(8, 9)
        seq_ids = range(1, 3)
        cam_ids = [8,0,2,7]

    if not exists(out_dir):
        makedirs(out_dir)

    out_path = join(out_dir, FLAGS.split + '_%04d.tfrecord')
    num_shards = FLAGS.train_shards

    # Load all data & shuffle it,,
    all_gt2ds, all_gt3ds, all_img_paths = [], [], []
    all_cams = []
    all_cam_info = read_camera(data_dir)

    for sub_id in sub_ids:
        for seq_id in seq_ids:
            print('collecting S%d, Seq%d' % (sub_id, seq_id))
            if (sub_id==4 and seq_id==2): # 4-2,(5-1,5-2?), 
                print("skipping S{} Seq{}".format(sub_id, seq_id))
                continue
            # Collect all data for each camera.
            # img_paths: N list
            # gt2ds/gt3ds: N x 17 x 2, N x 17 x 3
            img_paths, gt2ds, gt3ds, cams = get_all_data(
                data_dir, sub_id, seq_id, cam_ids, all_cam_info)

            all_img_paths.append(img_paths)
            all_gt2ds.append(gt2ds)
            all_gt3ds.append(gt3ds)
            all_cams.append(cams)

    all_gt2ds = np.vstack(all_gt2ds)
    all_gt3ds = np.vstack(all_gt3ds)
    all_cams = np.vstack(all_cams)
    all_img_paths = np.vstack(all_img_paths)
    assert (all_gt3ds.shape[0] == len(all_img_paths))
    # Now shuffle it all.
    shuffle_id = np.random.permutation(len(all_img_paths))
    all_img_paths = all_img_paths[shuffle_id]
    all_gt2ds = all_gt2ds[shuffle_id]
    all_gt3ds = all_gt3ds[shuffle_id]
    all_cams = all_cams[shuffle_id]

    save_to_tfrecord(out_path, all_img_paths, all_gt2ds, all_gt3ds, all_cams,
                     num_shards)


def main(unused_argv):
    print('Saving results to %s' % FLAGS.output_directory)

    if not exists(FLAGS.output_directory):
        makedirs(FLAGS.output_directory)

    if FLAGS.split == 'train' or FLAGS.split == 'val':
        is_train = FLAGS.split == 'train'
        process_mpi_inf_3dhp_train(
            FLAGS.data_directory, FLAGS.output_directory, is_train=is_train)
    else:
        print('Unknown split %s' % FLAGS.split)
        import ipdb
        ipdb.set_trace()


if __name__ == '__main__':
    tf.app.run()
