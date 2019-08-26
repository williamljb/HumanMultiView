"""
Demo of HumanMultiView.

Note that HumanMultiView requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px. 

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo --img_paths data/im1963.jpg
python -m demo --img_paths data/coco1.png

# On images, with openpose output
python -m demo --img_paths data/random.jpg --json_path data/random_keypoints.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf

#src_ortho
from src_ortho.util import renderer as vis_util
from src_ortho.util import image as img_util
from src_ortho.util import openpose as op_util
import src_ortho.config
from src_ortho.RunModel import RunModel
import os
from src_ortho.tf_smpl.batch_smpl import SMPL
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

flags.DEFINE_string('img_paths', 'data/im1963.jpg', 'Images to run, can be multi-view, separated by comma')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')
flags.DEFINE_integer('scale_size',224,'Scale size. Image will be scaled to this size and cropped at the center with 224x224 size.')


def measure(theta, oriverts):
    """
    theta: 85: 3+72+10
    verts:6890*3
    """
    SMPL_model = SMPL(config.smpl_model_path)
    tens_shape = tf.placeholder(tf.float32, shape=[1, 10])
    tens_pose = tf.zeros([1, 72])
    verts, res_joint, _ = SMPL_model(tens_shape, tens_pose, get_skin=True)
    sess = tf.Session(config=tf.ConfigProto(
        device_count = {'GPU': 0}
    ))
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    vts, gt3ds = sess.run([verts, res_joint], feed_dict={tens_shape: np.expand_dims(theta[-10:],0)})
    vts=vts[0]
    gt3ds=gt3ds[0]
    off = -3
    neck=[3799,337,155,300,216,426,3772,3721,3665]
    #neck=[3060,454,217,219,153,154,301,209,210,215,213,259,
    #    427,3167,3922,3771,3728,3727,3722,3724,3812,3666,3668,3730,3729,3945]
    arm=[5893,5111,4727]
    leg=[4313,4496,6825]
    chest=[6501,1257,690,3044,615,1425,740,2909,
        894,753,4243,4739,4252,4134,4686,4102,6491,4828]
    waist=[3507,1348,665,633,805,893,2917,6375,4376,4291,4147,4154,4813]
    hip=[3513,1209,1514,1457,3088,3140,3120,6544,6562,6512,4923,4986,4403]
    height=[3765,6859]
    node = lambda x, arr: arr[x+off]
    leng = lambda a, i, arr : np.sum((node(a[i], arr)-node(a[(i+1)%len(a)], arr))**2)**0.5
    total = lambda a, arr: np.sum([leng(a,i,arr) for i in range(len(a))])
    print('height:{}'.format(leng(height,0,vts)))
    print('neck:{}'.format(total(neck,vts)))
    print('arm:{}'.format(leng(arm,0,vts)+leng(arm,1,vts)))
    print('leg:{}'.format(leng(leg,0,vts)+leng(leg,1,vts)))
    print('chest:{}'.format(total(chest,vts)))
    print('waist:{}'.format(total(waist,vts)))
    print('hip:{}'.format(total(hip,vts)))
    f = open('/nfshomes/liangjb/Downloads/show.obj', 'w')
    for i in range(vts.shape[0]):
        f.write('v ')
        for j in range(vts.shape[1]):
            f.write('{} '.format(vts[i,j]))
        f.write('\n')


def visualize(input_imgs, imgs, proc_params, jointss, vertss, cams, view):
    """
    Renders the result in original image coordinate frame.
    """
    img = imgs[view]
    proc_param = proc_params[view]
    joints = jointss[view]
    verts = vertss[view][0]
    cam = cams[view][0]

    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    rend_img_overlay = renderer(
        vert_shifted, cam=cam_for_render, img=img)
    rend_img_overlay = vis_util.draw_skeleton(rend_img_overlay, joints_orig)
    rend_img = renderer(
        vert_shifted, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp1 = renderer.rotated(
        vert_shifted, 90, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp2 = renderer.rotated(
        vert_shifted, -90, cam=cam_for_render, img_size=img.shape[:2])

    import matplotlib.pyplot as plt
    # plt.ion()
    plt.figure(1, figsize=(10, 10))
    plt.clf()
    plt.subplot(231)
    plt.imshow(img)
    plt.title('input')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(input_imgs[0][view]/2+0.5)#skel_img)#
    plt.title('joint projection')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(rend_img_overlay)
    plt.title('3D Mesh overlay')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(rend_img)
    plt.title('3D mesh')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(rend_img_vp1)
    plt.title('diff vp')
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(rend_img_vp2)
    plt.title('diff vp')
    plt.axis('off')
    plt.draw()
    plt.show()
    plt.savefig('res.png')
    # io.imsave('ori{}.png'.format(view),img)
    io.imsave('ours{}.png'.format(view), rend_img_overlay)#rend_img[:,:,:3])#
    # import ipdb
    # ipdb.set_trace()
    # for i in range(126):
    #     k = 360.0/125.0*(i-4)
    #     rend_img_demo = renderer.rotated(vert_shifted, k, cam=cam_for_render, img_size=img.shape[:2])
    #     io.imsave('ours%03d.jpg'%(i), rend_img_demo[:,:,:3])
    fcs=[]
    with open("/nfshomes/liangjb/Downloads/faces.obj", 'r') as f:
        for i,lines in enumerate(f):
            fcs.append(lines)
    with open("/nfshomes/liangjb/Downloads/show.obj", 'w') as f:
        f.write("# OBJ file\n")
        for v in range(verts.shape[0]):
            f.write("v %.4f %.4f %.4f\n" % (verts[v,0],verts[v,1],verts[v,2]))
        for lines in fcs:
            f.write("{}".format(lines))


def preprocess_image(img_path, json_path=None, view=0):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    size=config.scale_size
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)
    # image center in (x,y)
    center = center[::-1]
    ori, _ = img_util.scale_and_crop(img, 1, center,
                                               config.img_size/size*np.max(img.shape[:2])*1)
    img = ori
    if json_path is None:
        if np.max(img.shape[:2]) != config.img_size:
            print('Resizing so the max image size is %d..' % config.img_size)
            # scale = (size / np.max(img.shape[:2]))
            scale = (config.img_size / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)
    # ori = np.maximum(ori.astype(int) - 25,0)
    # st=int(80./224*ori.shape[0])
    # en=int(144./224*ori.shape[1])
    # ori[st:en,st:en]=0
    io.imsave('ori{}.png'.format(view),ori)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    # crop = crop - 0.2
    # if view==0:
    #     crop[80:144,80:144]=-1

    return crop, proc_param, img

def clip(x):
    mini = 0.01
    maxi = 1
    return np.clip(x*(x>0),mini,maxi)*(x>0) + np.clip(x*(x<0),-maxi,-mini)*(x<0)


def main(img_paths, json_path=None):
    sess = tf.Session()
    paths = img_paths.split(',')
    num_views = len(paths)
    model = RunModel(config, 4, num_views, sess=sess)
    input_imgs, proc_params, imgs = [],[],[]

    for i,path in enumerate(paths):
        input_img, proc_param, img = preprocess_image(path, json_path, i)
        input_imgs.append(input_img)
        proc_params.append(proc_param)
        imgs.append(img)
    # Add batch dimension: 1 x D x D x 3
    # return
    input_imgs = np.expand_dims(np.array(input_imgs), 0)

    joints, verts, cams, joints3d, theta = model.predict(
        input_imgs, get_theta=True)
    measure(theta[0][0], verts[0][0]) # view, batch
    np.set_printoptions(precision=5,suppress=True)
    # print(theta[0][0][3:75].reshape((24,3)))
    # print(theta[0][0][-10:])
    # print(joints3d[0])
    # verts = clip((verts - joints3d[0][0,5,:]) / 100) + joints3d[0][0,5,:]

    for i in range(num_views):
       visualize(input_imgs, imgs, proc_params, joints, verts, cams, i)


if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    # config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    main(config.img_paths, config.json_path)
