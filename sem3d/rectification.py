import numpy as np
from .geometry import get_rotation_angles, rotate_img, rotate_point
from .ransac_skim import filter_outliers
from .kp import match_keypoints

def translation_alignment(img1, img2, q1, q2, x_margin=5):
    """
    """
    # X axis shifting
    x_shift = (q1 - q2).min(0).astype("int")[0]
    x_shift += x_margin * np.sign(x_shift)
    if x_shift > 0:
        img1 = img1[:,x_shift:]
        img2 = img2[:,:-x_shift]
        q1[:,0]-=x_shift
    elif x_shift < 0:
        img1 = img1[:,:x_shift]
        img2 = img2[:,-x_shift:]
        q2[:,0]+=x_shift
        
    y_shift = (q1 - q2).mean(0).astype("int")[1]
    
    # Y axis shifting
    if y_shift > 0:
        img1 = img1[y_shift:]
        img2 = img2[:-y_shift]
        q1[:,1]-=y_shift
    elif y_shift < 0:
        img1 = img1[:y_shift]
        img2 = img2[-y_shift:]
        q2[:,1]+=y_shift
    
    return img1, img2, q1, q2, x_shift, y_shift

def rotation_alignment(img1, img2, q1, q2):
    """
    """
    t1, t2 = get_rotation_angles(q1, q2)
    center1 = np.array(img1.shape)[::-1][None,:] // 2
    center2 = np.array(img2.shape)[::-1][None,:] // 2
    
    img1 = rotate_img(img1, t1)
    img2 = rotate_img(img2, t2)
    
    center1_ = np.array(img1.shape)[::-1][None,:] // 2
    center2_ = np.array(img2.shape)[::-1][None,:] // 2

    q1   = rotate_point(q1, t2, center1, center1_)
    q2   = rotate_point(q2, t1, center2, center2_)

    return img1, img2, q1, q2

def _rectify(img1, img2, q1, q2, x_margin=5):
    """
    """
    img1, img2, q1, q2 = rotation_alignment(img1, img2, q1, q2)
    img1, img2, q1, q2,_,_ = translation_alignment(img1, img2, q1, q2, 
                                                   x_margin=x_margin)
    return img1, img2, q1, q2

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

def rectify_pair(img1, img2, feat="sift", 
                 intensity_threshold=50,
                 residual_threshold=.5,
                 coef_threshold=.7,
                 dist_threshold=100,
                 min_samples=5,
                 x_margin=10):
    """
    """
    q1, q2 = get_filtered_kp(img1, img2, feat=feat, 
                 intensity_threshold = intensity_threshold,
                 residual_threshold = residual_threshold,
                 coef_threshold = coef_threshold,
                 dist_threshold = dist_threshold,
                 min_samples = min_samples)
        
    return _rectify(img1, img2, q1, q2, x_margin=x_margin)