""" Evaluates a trained model using placeholders. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from os.path import join, exists, dirname

from .tf_smpl import projection as proj_util
from .tf_smpl.batch_smpl import SMPL
from .models import get_encoder_fn_separate
import deepdish as dd

class RunModel(object):
    def __init__(self, config, num_views, num_true_views, sess=None):
        """
        Args:
          config
        """
        self.config = config
        self.load_path = config.load_path
        
        # Config + path
        if not config.load_path:
            raise Exception(
                "[!] You need to specify `load_path` to load a pretrained model"
            )
        if not exists(config.load_path + '.index'):
            print('%s doesnt exist..' % config.load_path)
            import ipdb
            ipdb.set_trace()

        # Data
        self.batch_size = config.batch_size
        self.img_size = config.img_size

        self.data_format = config.data_format
        self.smpl_model_path = config.smpl_model_path
        
        self.num_views = num_views
        self.num_true_views = num_true_views
        input_size = (self.batch_size, self.num_true_views, self.img_size, self.img_size, 3)
        self.images_pl = tf.placeholder(tf.float32, shape=input_size)

        # Model Settings
        self.num_stage = config.num_stage
        self.model_type = config.model_type
        self.joint_type = config.joint_type
        # Camera
        self.num_cam = 3
        self.proj_fn = proj_util.batch_orth_proj_idrot

        self.num_theta = 72        
        # Theta size: camera (3) + pose (24*3) + shape (10)
        self.total_params = self.num_cam + self.num_theta + 10

        self.smpl = SMPL(self.smpl_model_path, joint_type=self.joint_type)

        # self.theta0_pl = tf.placeholder_with_default(
        #     self.load_mean_param(), shape=[self.batch_size, self.total_params], name='theta0')
        # self.theta0_pl = tf.placeholder(tf.float32, shape=[None, self.total_params], name='theta0')

        self.build_test_model_ief()

        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess
        
        # Load data.
        self.saver = tf.train.Saver()
        self.prepare()      


    def load_mean_param(self):
        mean = np.zeros((1, self.total_params))
        # Initialize scale at 0.9
        mean[0, 0] = 0.9
        mean_path = join(
            dirname(self.smpl_model_path), 'neutral_smpl_mean_params.h5')
        mean_vals = dd.io.load(mean_path)

        mean_pose = mean_vals['pose']
        # Ignore the global rotation.
        mean_pose[:3] = 0.
        mean_shape = mean_vals['shape']

        # This initializes the global pose to be up-right when projected
        mean_pose[0] = np.pi

        mean[0, 3:] = np.hstack((mean_pose, mean_shape))
        self.mean_np = mean[0]
        mean = tf.constant(mean, tf.float32)
        self.mean_var = tf.Variable(
            mean, name="mean_param", dtype=tf.float32, trainable=False)
        init_mean = tf.tile(self.mean_var, [1, 1])
        return init_mean


    def build_test_model_ief(self):
        # Load mean value
        self.mean_var = self.load_mean_param()

        img_enc_fn, threed_enc_fn = get_encoder_fn_separate(self.model_type)
        # Extract image features.        
        self.img_feat, self.E_var = [],[]
        for i in range(self.num_views):
            tmp0, tmp1 = img_enc_fn(
            self.images_pl[:,i%self.num_true_views,:,:], is_training=False, reuse=(i>0))
            self.img_feat.append(tmp0)
            self.E_var.append(tmp1)
        self.E_var = self.E_var[0]
        
        # Start loop
        self.all_verts = []
        self.all_kps = []
        self.all_cams = []
        self.all_Js = []
        self.final_thetas = []
        tmp = tf.tile(self.mean_var, [self.batch_size, 1])
        theta_prev = [tmp for _ in range(self.num_views)]
        multiplier = 1
        for stage in np.arange(self.num_stage * multiplier):
            print('Iteration %d' % stage)
            for i in range(self.num_views):
                # ---- Compute outputs
                im1 = (i-1+self.num_views) % self.num_views
                theta_prev[i] = tf.concat([theta_prev[i][:, :self.num_cam+3], theta_prev[im1][:, self.num_cam+3:]], axis=1)
                state = tf.concat([self.img_feat[i], theta_prev[i]], 1)

                if i == 0 and stage == 0:
                    delta_theta, _ = threed_enc_fn(
                        state,
                        num_output=self.total_params,
                        is_training=False,
                        reuse=False)
                else:
                    delta_theta, _ = threed_enc_fn(
                        state,
                        num_output=self.total_params,
                        is_training=False,
                        reuse=True)

                # Compute new theta
                theta_here = theta_prev[i] + delta_theta
                theta_prev[i] = theta_here
        # use last pred as global pred
        for i in range(0, self.num_views):
            theta_here = tf.concat([theta_prev[i][:, :self.num_cam+3], theta_prev[0][:, self.num_cam+3:]], axis=1)
            cams = theta_here[:, :self.num_cam]                
            poses = theta_here[:, self.num_cam:(self.num_cam + self.num_theta)]
            shapes = theta_here[:, (self.num_cam + self.num_theta):]
            verts, Js, _ = self.smpl(shapes, poses, get_skin=True)
            # Project to 2D!
            pred_kp = self.proj_fn(Js, cams, name='proj_2d_stage%d' % i)
            self.all_verts.append(verts)
            self.all_kps.append(pred_kp)
            self.all_cams.append(cams)
            self.all_Js.append(Js)
            # save each theta.
            self.final_thetas.append(theta_here)



    def prepare(self):
        print('Restoring checkpoint %s..' % self.load_path)
        self.saver.restore(self.sess, self.load_path)        
        self.mean_value = self.sess.run(self.mean_var)
            
    def predict(self, images, get_theta=False):
        """
        images: num_batch, img_size, img_size, 3
        Preprocessed to range [-1, 1]
        """
        results = self.predict_dict(images)
        # thetas = results['theta']
        # print('ori cam\n', self.mean_np[:3])
        # print('ori rot\n', self.mean_np[3:6])
        # print('ori pose\n', self.mean_np[6:75])
        # print('ori shape\n', self.mean_np[75:])
        # for i in range(12):
        #     print('--------------------stage ',np.floor(i/4),' view ',i%4)
        #     cam_prev = thetas[i-4][0] if i>=4 else self.mean_np
        #     print('cam delta\n',(thetas[i][0]-cam_prev)[:3])
        #     print('rot delta\n',(thetas[i][0]-cam_prev)[3:6])
        #     param_prev = thetas[i-1][0] if i>=1 else self.mean_np
        #     print('pose delta\n', (thetas[i][0]-param_prev)[6:75])
        #     print('shape delta\n', (thetas[i][0]-param_prev)[75:])
        if get_theta:
            return results['joints'], results['verts'], results['cams'], results[
                'joints3d'], results['theta']
        else:
            return results['joints'], results['verts'], results['cams'], results[
                'joints3d']

    def predict_dict(self, images):
        """
        images: num_batch, img_size, img_size, 3
        Preprocessed to range [-1, 1]
        Runs the model with images.
        """
        feed_dict = {
            self.images_pl: images,
            # self.theta0_pl: self.mean_var,
        }
        pad = self.num_views-self.num_true_views
        st = self.num_views
        en = pad
        if en != 0:
            fetch_dict = {
                'joints': self.all_kps[-st:-en],
                'verts': self.all_verts[-st:-en],
                'cams': self.all_cams[-st:-en],
                'joints3d': self.all_Js[-st:-en],
                'theta': self.final_thetas[-st:-en],
            }
        else:
            fetch_dict = {
                'joints': self.all_kps[-st:],
                'verts': self.all_verts[-st:],
                'cams': self.all_cams[-st:],
                'joints3d': self.all_Js[-st:],
                'theta': self.final_thetas[-st:],
            }

        results = self.sess.run(fetch_dict, feed_dict)

        # Return joints in original image space.
        joints = np.array(results['joints'])
        results['joints'] = ((joints + 1) * 0.5) * self.img_size
        #print(results['theta'])
        #results['theta'] = results['theta'][0:1]
        return results
