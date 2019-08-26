"""
Utils for evaluation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy
from scipy.spatial.distance import directed_hausdorff


def compute_similarity_transform(S1, S2, verts1=None):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    if verts1 is None:
        verts1 = S1
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        verts1 = verts1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t
    verts1_hat = scale*R.dot(verts1) + t

    if transposed:
        S1_hat = S1_hat.T
        verts1_hat = verts1_hat.T

    return S1_hat, verts1_hat


def align_by_pelvis(joints, get_pelvis=False):
    """
    Assumes joints is 14 x 3 in LSP order.
    Then hips are: [3, 2]
    Takes mid point of these points, then subtracts it.
    """
    left_id = 3
    right_id = 2

    pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.
    if get_pelvis:
        return joints - np.expand_dims(pelvis, axis=0), pelvis
    else:
        return joints - np.expand_dims(pelvis, axis=0)


def compute_errors(gt3ds, preds, gtverts=None, pdverts=None, pfverts=None):
    """
    Gets MPJPE after pelvis alignment + MPJPE after Procrustes.
    Evaluates on the 14 common joints.
    Inputs:
      - gt3ds: N x views * 14 x 3
      - preds: N x views * 14 x 3
    """
        # TODO: compute hausdorff distance using pose and shape
        # 
        # directed_hausdorff(u, v)[0]
    errors, errors_pa, error_haus, error_pa_haus = [], [], [], []
    error_pf_haus = []
    pcks, aucs, pcks_pa, aucs_pa = [], [], [], []
    print(gt3ds.shape, preds.shape)
    assert gt3ds.shape == preds.shape
    num_views = gt3ds.shape[1]
    for i, (gt3d_, pred_) in enumerate(zip(gt3ds, preds)):
        for j in range(num_views):
            gt3d = gt3d_[j].reshape(-1, 3)
            # Root align.
            gt3d = align_by_pelvis(gt3d)
            pred3d = align_by_pelvis(pred_[j])

            joint_error = np.sqrt(np.sum((gt3d - pred3d)**2, axis=1))
            errors.append(np.mean(joint_error))
            if gtverts is not None:
                u = gtverts[i, j]
                v = pdverts[i, j]
                w = pfverts[i, j]
                haus_err = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
                pf_haus = max(directed_hausdorff(u, w)[0], directed_hausdorff(w, u)[0])
                error_haus.append(haus_err)
                error_pf_haus.append(pf_haus)
            else:
                v = None
                auc = 0
                for k in range(1, 151, 5):
                    pck = np.sum(joint_error <= k) / 14.0 * 100
                    auc += pck / 30.0
                aucs.append(auc)
                pcks.append(pck)

            # Get PA error.
            pred3d_sym, v = compute_similarity_transform(pred3d, gt3d, v)
            pa_error = np.sqrt(np.sum((gt3d - pred3d_sym)**2, axis=1))
            errors_pa.append(np.mean(pa_error))
            if gtverts is not None:
                u = gtverts[i, j]
                haus_pa_err = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
                error_pa_haus.append(haus_pa_err)
            else:
                auc = 0
                for k in range(1, 151, 5):
                    pck = np.sum(pa_error <= k) / 14.0 * 100
                    auc += pck / 30.0
                aucs_pa.append(auc)
                pcks_pa.append(pck)

    if gtverts is None:
        return errors, errors_pa, pcks, aucs, pcks_pa, aucs_pa
    else:
        return errors, errors_pa, error_haus, error_pa_haus, error_pf_haus
