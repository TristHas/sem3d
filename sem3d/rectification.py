import numpy as np
from scipy.ndimage.interpolation import shift

from .geometry import get_rotation_angles, rotate_point_img
from .ransac_skim import filter_outliers
from .kp import match_keypoints

def get_filtered_kp(img1, img2, feat="sift", 
                 intensity_threshold=50,
                 residual_threshold=.5,
                 coef_threshold=.7,
                 dist_threshold=100,
                 min_samples=5):
    """
    """
    q1, q2 = match_keypoints(img1, img2, 
                     filter_coef=coef_threshold, 
                     filter_dist=dist_threshold, 
                     filter_intesity=intensity_threshold)
    
    q1, q2 = filter_outliers(q1, q2, 
                             min_samples=min_samples, 
                             residual_threshold=residual_threshold)
    return q1, q2

def translation_alignment(img1, img2, q1, q2, x_margin=2):
    x_shift = (q1 - q2).max(0)[0] + x_margin
    y_shift = (q1 - q2).mean(0)[1]
    img1 = shift(img1, (-y_shift, -x_shift))
    q1 -= np.array([[x_shift, y_shift]])
    return img1, img2, q1, q2

def rotation_alignment(img1, img2, q1, q2):
    """
    """
    t1, t2 = get_rotation_angles(q1, q2)
    img1, q1 = rotate_point_img(img1, q1, t1)
    img2, q2 = rotate_point_img(img2, q2, t2)
    return img1, img2, q1, q2

def _rectify(img1, img2, q1, q2, x_margin=5):
    """
    """
    img1, img2, q1, q2 = rotation_alignment(img1, img2, q1, q2)
    img1, img2, q1, q2 = translation_alignment(img1, img2, q1, q2, 
                                               x_margin=x_margin)
    return img1, img2, q1, q2

def rectify(img1, img2, feat="sift", 
            intensity_threshold=50,
            residual_threshold=.5,
            coef_threshold=.7,
            dist_threshold=100,
            min_samples=5,
            x_margin=2):
    """
    """
    q1, q2 = get_filtered_kp(img1, img2, feat=feat, 
                 intensity_threshold = intensity_threshold,
                 residual_threshold = residual_threshold,
                 coef_threshold = coef_threshold,
                 dist_threshold = dist_threshold,
                 min_samples = min_samples)
        
    return _rectify(img1, img2, q1, q2, x_margin=x_margin)