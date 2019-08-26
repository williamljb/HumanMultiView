""" Convert MPI_INF_3DHP to TFRecords """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, exists
from os import makedirs, listdir

import numpy as np

import tensorflow as tf
from ..tf_smpl.batch_smpl import SMPL
import os.path as osp

from .common import convert_to_example_wmosh, ImageCoder, resize_img
from ..util import renderer as vis_util
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

tf.app.flags.DEFINE_string('data_directory', '/scratch1/storage/mpi_inf_3dhp/',
                           'data directory: top of mpi-inf-3dhp')
tf.app.flags.DEFINE_string('output_directory',
                           '/scratch1/projects/tf_datasets/mpi_inf_3dhp/',
                           'Output data directory')

tf.app.flags.DEFINE_string('split', 'train', 'train or trainval')
tf.app.flags.DEFINE_integer('train_shards', 500,
                            'Number of shards in training TFRecord files.')

curr_path = osp.dirname(osp.abspath(__file__))
model_dir = osp.join(curr_path, '../..', 'models')
if not osp.exists(model_dir):
    print('Fix path to models/')
    import ipdb
    ipdb.set_trace()
SMPL_MODEL_PATH = osp.join(model_dir, 'neutral_smpl_with_cocoplus_reg.pkl')
tf.app.flags.DEFINE_string('smpl_model_path', SMPL_MODEL_PATH,
                    'path to the neurtral smpl model')

FLAGS = tf.app.flags.FLAGS
MIN_VIS_PTS = 8  # This many points must be within the image.
DRESS_CODE=['dress','karate']
POSES = []

# To go to h36m joints:
# training joints have 28 joints
# test joints are 17 (H3.6M subset in CPM order)
sess = tf.Session()
SMPL_model = SMPL(FLAGS.smpl_model_path)
tens_shape = tf.placeholder(tf.float32, shape=[4, 10])
tens_pose = tf.placeholder(tf.float32, shape=[4, 72])
verts, res_joint, _ = SMPL_model(tens_shape, tens_pose, get_skin=True)

def get_all_data(base_dir, sub_id, seq_id, dup_id, num_frames_robe):
    img_dir = join(base_dir, '%d_%s' % (sub_id, DRESS_CODE[seq_id]))
    # frames, _, annot2, annot3 = read_mat(anno_path)

    all_gt2ds, all_gt3ds, all_img_paths = [], [], []
    all_cams, all_poses, all_shapes = [], [], []
    param_path = join(img_dir, 'posemot', 'conf.txt')
    f = open(param_path, 'r')
    shape = []
    cur_shape = []
    for i, line in enumerate(f):
        if i == 0:
            pose_chosen = int(line.split(':')[1])
        else:
            cur_shape.append(float(line))
    for cam_id in range(4):
        base_path = join(img_dir, 'img0%d' % dup_id, '%04d' + ('_%d0.jpg' % cam_id))
        jump_frames = 1
        num_frames = int(len(listdir(join(img_dir, 'img02'))) / 12) - jump_frames + 1
        if seq_id == 0:
            num_frames_robe = num_frames * 5 + pose_chosen
        gt2ds = -np.ones((num_frames, 14, 2))
        gt3ds = -np.ones((num_frames, 14, 3))
        # Convert N x 28 x . to N x 14 x 2, N x 14 x 3
        chosen_frames = [0] + [frame+jump_frames for frame in range(num_frames-1)]
        img_paths = [base_path % frame for frame in chosen_frames]
        pose = POSES[pose_chosen-1][:num_frames+jump_frames-1]
        pose = [pose[0]] + pose[jump_frames:]
        assert len(pose) == num_frames
        shape = [cur_shape for _ in range(num_frames)]
        if gt3ds.shape[0] != len(img_paths):
            print('Not same paths?')
            import ipdb
            ipdb.set_trace()
        all_gt2ds.append(gt2ds)
        all_gt3ds.append(gt3ds)
        cams = -np.ones((num_frames, 3))
        all_cams.append(cams)
        all_img_paths.append(np.array(img_paths).tolist())
        all_poses.append(pose)
        all_shapes.append(shape)

    all_gt2ds = np.transpose(np.array(all_gt2ds), [1,0,2,3])
    all_gt3ds = np.transpose(np.array(all_gt3ds), [1,0,2,3])
    all_cams = np.transpose(np.array(all_cams), [1,0,2])
    all_img_paths = np.transpose(np.array(all_img_paths), [1,0])
    all_poses = np.transpose(np.array(all_poses), [1,0,2])
    all_shapes = np.transpose(np.array(all_shapes), [1,0,2])
    return all_img_paths, all_gt2ds, all_gt3ds, all_cams, all_poses, all_shapes, num_frames_robe

imgsiz=512
cam_k = np.array(((imgsiz*35./32.0, 0.0, imgsiz/2),
        (0.0, imgsiz*35./32, imgsiz/2),
        (0.0, 0.0, 1.0)))
cam_RT = np.array(((1.0, 0.0, 0.0, -0.011766340583562851),
        (0.0, 0.9921149611473083, -0.12533271312713623, -0.27432456612586975),
        (0.0, 0.12533271312713623, 0.9921149611473083, 2.501513719558716)))
pose_init = None

def visualize_img(img, gt_kp, gt_vert):
    """
    Overlays gt_kp and pred_kp on img.
    Draws vert with text.
    Renderer is an instance of SMPLRenderer.
    """
    gt_kp = gt_kp[:14]
    renderer = vis_util.SMPLRenderer(
        img_size=224,
        face_path='/scratch1/hmr_multiview/src/tf_smpl/smpl_faces.npy')
    gt_vis = np.ones(14).astype(bool)
    # Fix a flength so i can render this with persp correct scale
    f = 5.
    cam_for_render = 0.5 * 224 * np.array([f, 1, 1])
    cam_t = np.array([0., 0., f/0.9])
    # Undo pre-processing.
    input_img = img
    rend_img = input_img

    # Draw skeleton
    gt_joint = 120+(gt_kp-120)/1.
    # print(gt_joint)
    # input_img = renderer(gt_vert + cam_t, cam_for_render, img=input_img) / 255.
    # input_img = renderer(gt_vert + cam_t, cam_for_render, img=input_img)
    img_with_gt = vis_util.draw_skeleton(
        input_img, gt_joint, draw_edges=False, vis=gt_vis)

    combined = np.hstack([img_with_gt/255., rend_img / 255.])

    import matplotlib.pyplot as plt
    # plt.ion()
    plt.imshow(combined)
    plt.show()
    # import ipdb; ipdb.set_trace()
    return combined

def align_by_torso(joints):
    """
    Assumes joints is N x 14 x 3 in LSP order.
    Then hips are: [3, 2]
    Takes mid point of these points, then subtracts it.
    """
    pelvis = (joints[:,2, :] + joints[:,3, :] + joints[:,8, :] + joints[:,9, :]) / 4.
    return -np.expand_dims(pelvis, axis=1)

def align_by_init(joints, torso):
    """
    Assumes joints is N x 14 x 3 in LSP order.
    Then hips are: [3, 2]
    Takes mid point of these points, then subtracts it.
    """
    # pelvis = (joints[2, :] + joints[3, :]) / 2.
    # pelvis_init = (joints_init[2, :] + joints_init[3, :]) / 2.
    # return joints - np.expand_dims(pelvis - pelvis_init, axis=0)
    return joints + torso

def add_to_tfrecord(im_paths,
                    gt2ds,
                    gt3ds,
                    cams,
                    coder,
                    writer,
                    pose,
                    shape,
                    model=None,
                    sub_path=None):
    """
    gt2ds is 4 * 14 x 2 (lsp order)
    gt3ds is 4 * 14 x 3
    cam is (4,3,)
    pose: (4,72)
    shape: (4,10)
    returns:
      success = 1 if this is a good image
      0 if most of the kps are outside the image
    """
    # Read image
    images, labels, heights, widths = [], [], [], []
    center_scaleds, scale_factorss = [], []
    start_pts, cam_scaleds = [], []
    pose[0,:3] = cv2.Rodrigues(np.matmul(
        cv2.Rodrigues(np.array([np.pi,0,0]))[0],
        cv2.Rodrigues(pose[0,:3])[0]
        ))[0][:,0]
    for i in range(4):
        # (x,z) -> (-z, x)
        if i > 0:
            pose[i,:] = pose[i-1,:]
            pose[i,:3] = cv2.Rodrigues(np.matmul(
                cv2.Rodrigues(np.array([0, -np.pi/2,0]))[0],
                cv2.Rodrigues(pose[i,:3])[0]
                ))[0][:,0]
    vts, gt3ds = sess.run([verts, res_joint], feed_dict={tens_shape: shape, tens_pose: pose})
    gt3ds_init = sess.run(res_joint, feed_dict={tens_shape: shape, tens_pose: pose_init})
    gt3ds_torso = align_by_torso(gt3ds_init)

    obj_path = im_paths[0]
    pos = [pos for pos, char in enumerate(obj_path) if char == '/']
    obj_num = int(obj_path[pos[-1]+1:pos[-1]+5])
    obj_path = obj_path[:pos[-2]+1] + 'posemot/{}.obj'.format(obj_num)
    # print(obj_path)
    with open(obj_path, 'r') as f:
        line = f.readline()
        if line[0] == '#':
            # print(obj_path)
            line = f.readline()
        ori_pt = [float(line.split(' ')[i]) for i in range(1, 4)]
    # print(ori_pt)
    ori_pt[1] = -ori_pt[1]
    ori_pt[2] = -ori_pt[2]

    for path, gt2d, gt3d, cam, vt, gt3d_torso in zip(im_paths, gt2ds, gt3ds, cams, vts, gt3ds_torso):
        if not exists(path):
            print('!!--%s doesnt exist! Skipping..--!!' % path)
            return False
        with tf.gfile.FastGFile(path, 'rb') as f:
            image_data = f.read()
        image = coder.decode_jpeg(coder.png_to_jpeg(image_data))
        assert image.shape[2] == 3

        gt3d_torso = (ori_pt - vt[0]).reshape(1,3)
        ori_pt[0], ori_pt[2] = -ori_pt[2], ori_pt[0]
        blender_cam = np.matmul(cam_k, cam_RT)
        np.set_printoptions(precision=5,suppress=True)
        gt2d = np.matmul(blender_cam[:,:3], np.transpose(align_by_init(gt3d, gt3d_torso))) + blender_cam[:,3:4]
        gt2d = np.transpose(gt2d)
        gt2d = gt2d[:,:2] / gt2d[:,2:3]

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
        # with open('tmp.jpg', 'w') as f:
        #     f.write(image_data_scaled)

        # gt3d_torso = (ori_pt - vt[0]).reshape(1,3)
        # ori_pt[0], ori_pt[2] = -ori_pt[2], ori_pt[0]
        # blender_cam = np.matmul(cam_k, cam_RT)
        # np.set_printoptions(precision=5,suppress=True)
        # comp_kp = np.matmul(blender_cam[:,:3], np.transpose(align_by_init(gt3d, gt3d_torso))) + blender_cam[:,3:4]
        # comp_kp = np.transpose(comp_kp)
        # comp_kp = comp_kp[:,:2] / comp_kp[:,2:3]
        # comp_kp[:, 0] *= scale_factors[0]
        # comp_kp[:, 1] *= scale_factors[1]
        # comp_kp[:, 0] -= start_pt[0]
        # comp_kp[:, 1] -= start_pt[1]
        # print(comp_kp[13])
        fcs = []
        vt = align_by_init(vt, gt3d_torso)
        # with open("/nfshomes/liangjb/Downloads/faces.obj", 'r') as f:
        #     for i,lines in enumerate(f):
        #         fcs.append(lines)
        # with open("/nfshomes/liangjb/Downloads/show.obj", 'w') as f:
        #     f.write("# OBJ file\n")
        #     for v in range(vt.shape[0]):
        #         f.write("v %.4f %.4f %.4f\n" % (vt[v,0],-vt[v,1],-vt[v,2]))
        #     for lines in fcs:
        #         f.write("{}".format(lines))
        # print(path)
        # visualize_img(image_scaled, joints_scaled, vt)
        label = np.vstack([joints_scaled.T, np.ones((1, joints_scaled.shape[0]))]) # visibility = 0 to block 2d loss
        images.append(image_data_scaled)
        labels.append(label)
        heights.append(height)
        widths.append(width)
        center_scaleds.append(center_scaled)
        scale_factorss.append(scale_factors)
        start_pts.append(start_pt)
        cam_scaleds.append(cam_scaled)

    # pose and shape is not existent.
    example = convert_to_example_wmosh(
        images, im_paths, heights, widths, labels, center_scaleds, gt3ds,
        pose, shape, scale_factorss, start_pts, cam_scaleds)
    writer.write(example.SerializeToString())

    return True


def save_to_tfrecord(out_name, im_paths, gt2ds, gt3ds, cams, num_shards, all_poses, all_shapes):
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
                                          cams[i], coder, writer, all_poses[i], all_shapes[i])
                i += 1
                if success:
                    j += 1
                else:
                    num_bad += 1

        fidx += 1

    print('Done, wrote to %s, num skipped %d' % (out_name, num_bad))


def process_synthetic_train(data_dir, out_dir, is_train=False):
    if is_train:
        out_dir = join(out_dir, 'train')
        print('!train set!')
        sub_ids = range(0, 90)  # No S8!
        dress_ids = range(0, 2)
    else:  # Full set!!
        out_dir = join(out_dir, 'val')
        print('doing the full train-val set!')
        sub_ids = range(90, 100)
        dress_ids = range(0, 2)

    if not exists(out_dir):
        makedirs(out_dir)

    out_path = join(out_dir, FLAGS.split + '_%04d.tfrecord')
    num_shards = FLAGS.train_shards

    # Load all data & shuffle it,,
    all_gt2ds, all_gt3ds, all_img_paths = [], [], []
    all_cams, all_poses, all_shapes = [], [], []

    for pose_num in range(1,6):
        f = open(join(data_dir, 'newposes/pose%d.txt'%pose_num),'r');
        pose = []
        for i,line in enumerate(f):
            x=[]
            for y in line.split(','):
                if y=='' or y=='\n':
                    continue
                x.append(float(y))
            pose.append(x)
        POSES.append(pose)
    num_frames_robe = 0

    for sub_id in sub_ids:
        for seq_id in dress_ids:
            for dup_id in range(2, 4):
                print('collecting S%d in %s dup%d' % (sub_id, DRESS_CODE[seq_id], dup_id))
                # Collect all data for each camera.
                # img_paths: N list
                # gt2ds/gt3ds: N x 17 x 2, N x 17 x 3
                # poses: N*4*72
                # shapes: N*4*10
                img_paths, gt2ds, gt3ds, cams, poses, shapes, num_frames_robe = get_all_data(
                    data_dir, sub_id, seq_id, dup_id, num_frames_robe)
                global pose_init
                pose_init = poses[0]

                all_img_paths.append(img_paths)
                all_gt2ds.append(gt2ds)
                all_gt3ds.append(gt3ds)
                all_cams.append(cams)
                all_poses.append(poses)
                all_shapes.append(shapes)
    pose_init[0,:3] = cv2.Rodrigues(np.matmul(
        cv2.Rodrigues(np.array([np.pi,0,0]))[0],
        cv2.Rodrigues(pose_init[0,:3])[0]
        ))[0][:,0]
    for i in range(4):
        # (x,z) -> (-z, x)
        if i > 0:
            pose_init[i,:] = pose_init[i-1,:]
            pose_init[i,:3] = cv2.Rodrigues(np.matmul(
                cv2.Rodrigues(np.array([0, -np.pi/2,0]))[0],
                cv2.Rodrigues(pose_init[i,:3])[0]
                ))[0][:,0]

    all_gt2ds = np.vstack(all_gt2ds)
    all_gt3ds = np.vstack(all_gt3ds)
    all_cams = np.vstack(all_cams)
    all_img_paths = np.vstack(all_img_paths)
    all_poses = np.vstack(all_poses)
    all_shapes = np.vstack(all_shapes)
    assert (all_poses.shape[0] == len(all_img_paths))
    # Now shuffle it all.
    shuffle_id = np.random.permutation(len(all_img_paths))
    all_img_paths = all_img_paths[shuffle_id]
    all_gt2ds = all_gt2ds[shuffle_id]
    all_gt3ds = all_gt3ds[shuffle_id]
    all_cams = all_cams[shuffle_id]
    all_poses = all_poses[shuffle_id]
    all_shapes = all_shapes[shuffle_id]

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    save_to_tfrecord(out_path, all_img_paths, all_gt2ds, all_gt3ds, all_cams,
                     num_shards, all_poses, all_shapes)


def main(unused_argv):
    print('Saving results to %s' % FLAGS.output_directory)

    if not exists(FLAGS.output_directory):
        makedirs(FLAGS.output_directory)

    if FLAGS.split == 'train' or FLAGS.split == 'val':
        is_train = FLAGS.split == 'train'
        process_synthetic_train(
            FLAGS.data_directory, FLAGS.output_directory, is_train=is_train)
    else:
        print('Unknown split %s' % FLAGS.split)
        import ipdb
        ipdb.set_trace()


if __name__ == '__main__':
    tf.app.run()
