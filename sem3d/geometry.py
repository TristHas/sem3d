import numpy as np
from skimage.transform import rotate

def fundamental_matrix(A,B):
    X = np.concatenate([A, B], axis=1)
    mean = X.mean(0, keepdims=True)
    
    u, s, vh = np.linalg.svd(X-mean, full_matrices=False)
    np.linalg.norm(X.dot(vh.T), axis=0)
    params = vh[-1]
    
    a, b, c, d = params
    e = -mean.dot(params) 
    return a, b, c, d, e

def rad_to_deg(x):
    return x*180 / np.pi

def deg_to_rad(x):
    return x*np.pi / 180

def coef_to_angle(a,b):
    """
    """
    return rad_to_deg(np.arctan(-a/b))
    
def get_rotation_angles(q1, q2):
    """
        returns t1,t2 in degrees
    """
    a,b,c,d,e = fundamental_matrix(q1, q2)
    t1 = coef_to_angle(a, b) 
    t2 = coef_to_angle(c, d)
    return t1, t2

def rotation_mat(t):
    """
        Get rotation matrix corresponding to counter-clockwise rotation t
    """
    t = deg_to_rad(t)
    return np.array([[np.cos(t), -np.sin(t)],
                     [np.sin(t), np.cos(t)]]).T
    
def rotate_img(img, t):
    """
        Rotate points by a counter-clockwise rotation t
    """
    print(f"Rotating image by {t}")
    return rotate(img, t, resize=False,
                  preserve_range=True).astype('uint8')

def rotate_point(q, t, center=0, center_=0):
    """
        Rotate points by a counter-clockwise rotation t
    """
    print(f"Rotating points by {t}")
    q = q-center
    q = (rotation_mat(t) @ q.T).T
    q = center_ + q
    return q



###
### Not good file, will need some refactoring
###

def center(img1, img2, q1, q2):
    """
    """
    shift = (q1.mean(0) - q2.mean(0)).astype("int")
    x_shift, y_shift = shift
    if x_shift > 0:
        img1 = img1[:,x_shift:]
        img2 = img2[:,:-x_shift]
        q1[:,0]-=x_shift
    elif x_shift < 0:
        img1 = img1[:,:x_shift]
        img2 = img2[:,-x_shift:]
        q2[:,0]+=x_shift
        
    if y_shift > 0:
        img1 = img1[y_shift:]
        img2 = img2[:-y_shift]
        q1[:,1]-=y_shift
    elif y_shift < 0:
        img1 = img1[:y_shift]
        img2 = img2[-y_shift:]
        q2[:,1]+=y_shift
    
    return img1, img2, q1, q2

def crop_patch(img, corner=(200, 400), size=(400, 50)):
    x,y = corner
    w,h = size
    return img[y:y+h,  x:x+w]