import numpy as np
import cv2

detectors = {
    "akaze": cv2.AKAZE_create(),
    "sift" : cv2.xfeatures2d.SIFT_create()
}

matchers = {
    "akaze": cv2.BFMatcher(cv2.NORM_HAMMING),
    "sift" : cv2.FlannBasedMatcher({"algorithm":0, "trees":5}, {"checks":50})
}

def match_keypoints(img1, img2, feat="sift", 
                    filter_coef=.7, filter_dist=50, filter_intesity=20):
    """
    """
    kp1, des1 = detectors[feat].detectAndCompute(img1, None)
    kp2, des2 = detectors[feat].detectAndCompute(img2, None)
    matches = matchers[feat].knnMatch(des1, des2, k=2)
    nkp = len(matches)
    
    print(f"{nkp} keypoints matched")
    if filter_coef:
        matches = _filter_matches(matches, filter_coef)
        nkp2 = len(matches)
        print(f"{nkp-nkp2} keypoints filtered by coef ({filter_coef}). {nkp2} remaings")
        nkp = nkp2
    
    q1, q2 = _matches_to_np(kp1, kp2, matches)
    if filter_dist:
        q1,q2 = filter_by_dist(q1, q2, filter_dist)
        nkp2 = q1.shape[0]
        print(f"{nkp-nkp2} keypoints filtered by distance ({filter_dist}). {nkp2} remaings")
        nkp = nkp2
        
    if filter_intesity:
        q1,q2 = filter_by_intensity(img1, img2, q1, q2, filter_intesity)
        nkp2 = q1.shape[0]
        print(f"{nkp-nkp2} keypoints filtered by intensity ({filter_intesity}). {nkp2} remaings")
        
    return q1, q2

def _filter_matches(matches, coef=.7):
    """
    """
    outs = []
    for i,(m,n) in enumerate(matches):
        if m.distance < coef * n.distance:
            outs.append((m.queryIdx, m.trainIdx))
    return np.array(outs)

def filter_by_dist(q1, q2, max_dist=100):
    """
    """
    msk = np.linalg.norm(_center(q1) - _center(q2), axis=1) < max_dist
    return q1[msk], q2[msk]

def filter_by_intensity(img1, img2, q1, q2, thresh=25):
    """
    """
    x1,y1 = q1.astype(int).T
    x2,y2 = q2.astype(int).T
    msk1 = img1[y1,x1] > thresh
    msk2 = img2[y2,x2] > thresh
    msk = np.logical_and(msk1, msk2)
    return q1[msk], q2[msk]

def _center(x):
    """
    """
    return x-x.mean(0, keepdims=True)

def _matches_to_np(kp1, kp2, matches):
    """
    """
    KP1 = np.array(list(map(lambda x:x.pt, kp1)))
    KP2 = np.array(list(map(lambda x:x.pt, kp2)))
    return KP1[matches[:,0]], KP2[matches[:,1]]

###
### Old
###
def get_features(imgs, feat="sift"):
    """
    """
    kpdes = {}
    det,_ = get_matcher(feat)
    return {key:det.detectAndCompute(img,None) \
                for key,img in imgs.items()}

def match_features(feat1, feat2, feats="sift"):
    """
    """
    kp1, des1 = feat1
    kp2, des2 = feat2
    _, matcher = get_matcher(feat)
    # Match points of kp1 with points of kp2
    matches = matcher.knnMatch(des1, des2, k=2)
    # (q -> 2 most similar points in q')
    matches = matcher.knnMatch(des1, des2, k=2)
    matches = filter_matches(matches)
    q1,q2 = matches_to_np(kp1, kp2, matches)

    newmatches = filter_matches(matches)
    # If most similar q' is much better than the second most similar
    # Then select it as a good match
    
    # Return good (q,q') as numpy arrays
    KP1 = np.array(list(map(lambda x:x.pt, kp1)))
    KP2 = np.array(list(map(lambda x:x.pt, kp2)))
    q1, q2 = KP1[newmatches[:,0]], KP2[newmatches[:,1]]
    q1_idx = newmatches[:,0]
    
    return q1_idx, q1, q2