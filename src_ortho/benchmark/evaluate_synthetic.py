"""
Evaluation of H3.6M. 

Sample call from hmr:
python -m src.benchmark.evaluate_mpi3dhp --batch_size=50 --load_path=<model_to_eval>
python -m src.benchmark.evaluate_h36m --batch_size=500 --load_path=/home/kanazawa/projects/hmr_v2/models/model.ckpt-667589
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from absl import flags
import numpy as np
import deepdish as dd
from time import time
from os.path import exists, join, expanduser, split
from os import makedirs

import tensorflow as tf

from src_ortho.config import get_config
from ..util import renderer as vis_util
from ..RunModel import RunModel
from .eval_util import compute_errors
from ..datasets.common import read_images_from_tfrecords

kPredDir = '/tmp/hmr_output'
# Change to where you saved your tfrecords
kTFDataDir = '/scratch1/williamljb/hmr_multiview/tf_datasets/synthetic'

flags.DEFINE_string('pred_dir', kPredDir,
                           'where to save model output of h36m')
flags.DEFINE_string('tfh36m_dir', kTFDataDir,
                           'data dir: top of h36m in tf_records')
flags.DEFINE_integer(
    'protocol', 1,
    'If 2, then only frontal cam (3) and trial 1, if 1, then all camera & trials'
)
flags.DEFINE_boolean(
    'vis', False, 'If true, visualizes the best and worst 30 results.')
flags.DEFINE_boolean('sv', False, 'If true, input as single view data')

model = None
sess = None
# For visualization.
renderer = None
extreme_errors, contents = [], []
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# -- Draw Utils ---
def draw_content(content, config):
    global renderer
    input_img = content['image']
    vert = content['vert']
    joint = content['joint']
    cam = content['cam']
    img_size = config.img_size
    # Undo preprocessing
    img = (input_img + 1) * 0.5 * 255
    tz = renderer.flength / (0.5 * img_size * cam[0])
    trans = np.hstack([cam[1:], tz])
    vert_shifted = vert + trans
    # Draw
    skel_img = vis_util.draw_skeleton(img, joint)
    rend_img = renderer(vert_shifted, img_size=(img_size, img_size))
    another_vp = renderer.rotated(vert_shifted, 90, img_size=(img_size, img_size), do_alpha=False)
    another_vp = vis_util.draw_text(another_vp, {"diff_viewpoint": 90})

    tog0 = np.hstack((img, rend_img))
    tog1 = np.hstack((skel_img, another_vp))

    all_img = np.vstack((tog0, tog1)).astype(np.uint8)
    # import matplotlib.pyplot as plt
    # plt.ion()
    # plt.figure(1)
    # plt.clf()
    # plt.imshow(all_img)
    # plt.axis('off')
    # import ipdb; ipdb.set_trace()

    return all_img

        
# -- Utils ---
def get_pred_dir(base_dir, load_path):
    bpath, checkpt_name = split(load_path)
    bpath, model_name = split(bpath)
    _, log_name = split(bpath)
    pred_dir = join(
        expanduser(base_dir), '-'.join([log_name, model_name, checkpt_name]))
    return pred_dir


def get_seqs():
    all_pairs = [('%04d'%i) for i in range(8)]

    return all_pairs


# -- Core: ---
def get_data(seq_name, config):
    """
    Read preprocessed image from tfrecords.
    """
    global sess
    if sess is None:
        sess = tf.Session()

    tf_path = join(
        expanduser(config.tfh36m_dir), 'val', 'val_' + seq_name + '.tfrecord')
    images, kps, gt3ds, poses, shapes = read_images_from_tfrecords(
        tf_path, 4, img_size=config.img_size, sess=sess)
    view = 1 if config.sv else 4
    ims = [-1,view]+list(images.shape[2:])
    gs = [-1,view]+list(gt3ds.shape[2:])
    ps = [-1,view]+list(poses.shape[2:])
    ss = [-1,view]+list(shapes.shape[2:])

    return images.reshape(ims), gt3ds.reshape(gs), poses.reshape(ps), shapes.reshape(ss)


def run_model(images, config, gtpose, gtshape):
    """
    Runs trained model to get predictions on each seq.
    """
    global model, sess, tens_shape, tens_pose, tens_verts
    # Setup model with this config.
    num_views = 1 if config.sv else 4
    batch_size = config.batch_size
    N = len(images)
    if model is None:
        model = RunModel(config, 4, num_views, sess=sess)
        tens_shape = tf.placeholder(tf.float32, shape=[N, num_views, 10])
        tens_pose = tf.placeholder(tf.float32, shape=[N, num_views, 72])
        tens_verts, _, _ = model.smpl(tf.reshape(tens_shape, [-1, 10]), tf.reshape(tens_pose, [-1, 72]), 
            get_skin=True)
        tens_verts = tf.reshape(tens_verts, [N, num_views, -1, 3])
        print(tens_pose)

    tens_len = tens_shape.shape[0].value
    if N < tens_len:
        gtshape = np.vstack([gtshape, np.zeros((tens_len - N, num_views, 10))])
        gtpose = np.vstack([gtpose, np.zeros((tens_len - N, num_views, 72))])
    gtverts = sess.run(tens_verts, feed_dict={tens_shape:gtshape, tens_pose:gtpose})
    gtverts = gtverts[:N]

    all_joints, all_verts, all_cams, all_joints3d, all_thetas = [], [], [], [], []
    all_pfverts = []

    # Batch + preprocess..
    num_total_batches = int(np.ceil(float(N) / batch_size))
    for b in xrange(num_total_batches):
        print('Batch %d/%d' % (b, num_total_batches))
        start_ind = b * batch_size
        end_ind = (b + 1) * batch_size
        images_here = images[start_ind:end_ind]

        if end_ind > N:
            end_ind = N
            # Need to pad dummy bc batch size is not dynamic,,
            num_here = images_here.shape[0]
            images_wdummy = np.vstack([
                images_here,
                np.zeros((batch_size - num_here, num_views, config.img_size,
                          config.img_size, 3))
            ])
            joints, verts, cams, joints3d, thetas = model.predict(
                images_wdummy, get_theta=True)
            joints = [joint[:num_here] for joint in joints]
            verts = [vert[:num_here] for vert in verts]
            cams = [cam[:num_here] for cam in cams]
            joints3d = [joints3[:num_here] for joints3 in joints3d]
            thetas = [theta[:num_here] for theta in thetas]
        else:
            joints, verts, cams, joints3d, thetas = model.predict(
                images_here, get_theta=True)

        all_joints.append(joints)
        all_verts.append(verts)
        all_cams.append(cams)
        all_joints3d.append(joints3d)
        all_thetas.append(thetas)

    # view * B * size
    preds = {
        'verts': np.hstack(all_verts),
        'cams': np.hstack(all_cams),
        'joints': np.hstack(all_joints),
        'joints3d': np.hstack(all_joints3d),
        'thetas': np.hstack(all_thetas)
    }
    predshape = np.array(preds['thetas'])[:,:,-10:].transpose((1,0,2))
    if N < tens_len:
        predshape = np.vstack([predshape, np.zeros((tens_len - N, num_views, 10))])
    pfverts = sess.run(tens_verts, feed_dict={tens_shape:predshape,
        tens_pose:gtpose})[:N]

    # Check output.
    # for i in xrange(10):
    #     content = {
    #         'vert': preds['verts'][i],
    #         'joint': preds['joints'][i],
    #         'image': images[i],
    #         'cam': preds['cams'][i],
    #     }
    #     rend_img = draw_content(content, config)

    return preds, gtverts, pfverts


def add_visuals(errors, results, images):
    global extreme_errors, contents
    # Record extreme ones
    sort_inds = np.argsort(errors)[::-1]
    # Save top/worst 10.
    for i in xrange(4):
        ind = sort_inds[i]
        indd4 = int(np.floor(ind/4))
        content = {
            'vert': results['verts'][ind%4][indd4],
            'joint': results['joints'][ind%4][indd4],
            'image': images[indd4][ind%4],
            'cam': results['cams'][ind%4][indd4],
        }
        extreme_errors.append(errors[ind])
        contents.append(content)
        # Save best too.
        best_ind = sort_inds[-(i + 1)]
        best_indd4 = int(np.floor(best_ind/4))
        content = {
            'vert': results['verts'][best_ind%4][best_indd4],
            'joint': results['joints'][best_ind%4][best_indd4],
            'image': images[best_indd4][best_ind%4],
            'cam': results['cams'][best_ind%4][best_indd4],
        }
        extreme_errors.append(errors[best_ind])
        contents.append(content)


def evaluate_sequence(seq_info, pred_dir):
    print('%s' % (seq_info))

    save_path = join(pred_dir, seq_info + '_pred.h5')
    if exists(save_path):
        results = dd.io.load(save_path)
        errors = results['errors']
        errors_pa = results['errors_pa']
        error_haus = results['error_haus']
        error_pa_haus = results['error_pa_haus']
        error_pf_haus = results['error_pf_haus'] #pose fixed haus
        if config.vis:
            # Need to load images too..
            images, gt3ds, poses, shapes = get_data(seq_info, config)
    else:
        # Run the model!
        t0 = time()
        images, gt3ds, poses, shapes = get_data(seq_info, config)

        results, gtverts, pfverts = run_model(images, config, poses, shapes)
        t1 = time()
        print('Took %g sec for %d imgs' % (t1 - t0, len(results['verts'][0])))

        # Evaluate!
        # Joints 3D is COCOplus format now. First 14 is H36M joints
        pred3ds = np.array(results['joints3d'])[:, :, :14, :].transpose((1, 0, 2, 3))
        pdverts = np.array(results['verts']).transpose((1, 0, 2, 3))
        # Convert to mm!
        errors, errors_pa, error_haus, error_pa_haus, error_pf_haus = compute_errors(
            gt3ds * 1000., pred3ds * 1000., gtverts * 1000., pdverts * 1000., pfverts*1000.)
        # for i in range(int(len(errors) / 4)):
        #     for j in range(4):
        #         errors[i*4+j] = np.min(errors[i*4:i*4+4])
        #         errors_pa[i*4+j] = np.min(errors_pa[i*4:i*4+4])
        #         error_haus[i*4+j] = np.min(error_haus[i*4:i*4+4])
        #         error_pa_haus[i*4+j] = np.min(error_pa_haus[i*4:i*4+4])

        results['errors'] = errors
        results['errors_pa'] = errors_pa
        results['error_haus'] = error_haus
        results['error_pa_haus'] = error_pa_haus
        results['error_pf_haus'] = error_pf_haus
        # Save results
        dd.io.save(save_path, results)

    if config.vis:
        add_visuals(errors, results, images)

    return errors, errors_pa, error_haus, error_pa_haus, error_pf_haus


def main(config):
    # Figure out the save name.
    pred_dir = get_pred_dir(config.pred_dir, config.load_path)
    pred_dir += '_synthetic'
    if config.sv:
        pred_dir += 'sv'
    print('\n***\nsaving predictions in %s\n***\n' % pred_dir)

    if not exists(pred_dir):
        makedirs(pred_dir)

    if config.vis:
        global renderer
        # Bad impl with global..
        global extreme_errors, contents
        renderer = vis_util.SMPLRenderer(
            img_size=config.img_size, face_path=config.smpl_face_path)

    all_pairs = get_seqs()

    all_errors = {}
    all_errors_pa = {}
    raw_errors, raw_errors_pa, raw_errors_pf = [], [], []
    raw_errors_haus, raw_errors_pa_haus, raw_errors_pf_haus = [], [], []
    for itr, seq_info in enumerate(all_pairs):
        print('%d/%d' % (itr, len(all_pairs)))
        errors, errors_pa, error_haus, error_pa_haus, error_pf_haus = evaluate_sequence(seq_info, pred_dir)
        mean_error = np.mean(errors)
        mean_error_pa = np.mean(errors_pa)
        mean_error_haus = np.mean(error_haus)
        mean_error_pa_haus = np.mean(error_pa_haus)
        mean_error_pf_haus = np.mean(error_pf_haus)
        med_error = np.median(errors)
        raw_errors.append(errors)
        raw_errors_pa.append(errors_pa)
        raw_errors_haus.append(error_haus)
        raw_errors_pa_haus.append(error_pa_haus)
        raw_errors_pf_haus.append(error_pf_haus)
        print('====================')
        print('mean error: %g, median: %g, PA mean: %g, haus: %d, PA haus: %g' % 
            (mean_error, med_error, mean_error_pa, mean_error_haus, mean_error_pa_haus))
        raws = np.hstack(raw_errors)
        raws_pa = np.hstack(raw_errors_pa)
        raws_haus = np.hstack(raw_errors_haus)
        raws_pa_haus = np.hstack(raw_errors_pa_haus)
        raws_pf_haus = np.hstack(raw_errors_pf_haus)
        print('Running average - mean: %g, median: %g' % (np.mean(raws),
                                                          np.median(raws)))
        print('Running average - PA mean: %g, median: %g' %
              (np.mean(raws_pa), np.median(raws_pa)))
        print('Running average - haus mean: %g, median: %g' % (np.mean(raws_haus),
                                                          np.median(raws_haus)))
        print('Running average - PA haus mean: %g, median: %g' %
              (np.mean(raws_pa_haus), np.median(raws_pa_haus)))
        print('Running average - PF haus mean: %g, median: %g' %
              (np.mean(raws_pf_haus), np.median(raws_pf_haus)))
        print('====================')

    print('--for %s--' % config.load_path)

    err_pa = np.hstack(raw_errors_pa)
    MPJPE = np.mean(np.hstack(raw_errors))
    PA_MPJPE = np.mean(err_pa)
    print('Average error over all joints 3d: %g, PA: %g' % (MPJPE, PA_MPJPE))

    err = np.hstack(raw_errors)
    median = np.median(np.hstack(raw_errors))
    pa_median = np.median(np.hstack(err_pa))
    print(
        'Percentiles 90th: %.1f 70th: %.1f 50th: %.1f 30th: %.1f 10th: %.1f' %
        (np.percentile(err, 90), np.percentile(err, 70),
         np.percentile(err, 50), np.percentile(err, 30),
         np.percentile(err, 10)))

    print('MPJPE: %.2f, PA-MPJPE: %.2f, Median: %.2f, PA-Median: %.2f' %
          (MPJPE, PA_MPJPE, median, pa_median))

    if config.vis:
        global extreme_errors, contents
        import matplotlib.pyplot as plt
        # plt.ion()
        plt.figure(1)
        plt.clf()
        sort_inds = np.argsort(extreme_errors)[::-1]
        for i in xrange(30):
            bad_ind = sort_inds[i]
            bad_error = extreme_errors[bad_ind]
            bad_img = draw_content(contents[bad_ind], config)
            plt.figure(1)
            plt.clf()
            plt.imshow(bad_img)
            plt.axis('off')
            plt.title('%d-th worst, mean error %.2fmm' % (i, bad_error))

            good_ind = sort_inds[-(i+1)]
            good_error = extreme_errors[good_ind]
            good_img = draw_content(contents[good_ind], config)
            plt.figure(2)
            plt.clf()
            plt.imshow(good_img)
            plt.axis('off')
            plt.title('%d-th best, mean error %.2fmm' % (i, good_error))
            plt.draw()
            plt.show()
            # import ipdb; ipdb.set_trace()

    

if __name__ == '__main__':
    config = get_config()
    if not config.load_path:
        raise Exception('Must specify a model to use to predict!')
    if 'model.ckpt' not in config.load_path:
        raise Exception('Must specify a model checkpoint!')
    main(config)
