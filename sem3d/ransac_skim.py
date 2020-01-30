import numpy as np
from skimage.measure.fit import *
from skimage.measure.fit import _dynamic_max_trials

from .geometry import fundamental_matrix, coef_to_angle

def ransac(data, model_class, min_samples, residual_threshold,
           is_data_valid=None, is_model_valid=None,
           max_trials=100, stop_sample_num=np.inf, stop_residuals_sum=0,
           stop_probability=1, random_state=None, initial_inliers=None):
    best_model = None
    best_inlier_num = 0
    best_inlier_residuals_sum = np.inf
    best_inliers = None
    random_state = check_random_state(random_state)
    num_samples = len(data[0])

    for num_trials in range(max_trials):
        # choose random sample
        spl_idxs = random_state.choice(num_samples, min_samples, replace=False)
        samples = [d[spl_idxs] for d in data]
        
        sample_model = model_class()
        success = sample_model.estimate(samples)

        sample_model_residuals    = np.abs(sample_model.residuals(data))
        sample_model_residuals_sum = np.sum(sample_model_residuals ** 2)
        sample_model_inliers      = sample_model_residuals < residual_threshold
        sample_inlier_num = np.sum(sample_model_inliers)
        
        if (
            sample_inlier_num > best_inlier_num
            or (sample_inlier_num == best_inlier_num
                and sample_model_residuals_sum < best_inlier_residuals_sum)
        ):
            best_model = sample_model
            best_inlier_num = sample_inlier_num
            best_inlier_residuals_sum = sample_model_residuals_sum
            best_inliers = sample_model_inliers
            dynamic_max_trials = _dynamic_max_trials(best_inlier_num,
                                                     num_samples,
                                                     min_samples,
                                                     stop_probability)
            if (best_inlier_num >= stop_sample_num
                or best_inlier_residuals_sum <= stop_residuals_sum
                or num_trials >= dynamic_max_trials):
                break

    # estimate final model using all inliers
    if best_inliers is not None:
        # select inliers for each data array
        data_inliers =  [d[best_inliers] for d in data]
        best_model.estimate(data_inliers)

    return best_model, best_inliers

class FundamentalMatrix():
    def __init__(self):
        self.ab = np.zeros(2)
        self.cd = np.zeros(2)
        self.e = 0
        
    def estimate(self, data):
        """
        """
        q1, q2 = data
        a,b,c,d,e  = fundamental_matrix(q1, q2)
        self.ab[:] = a, b
        self.cd[:] = c, d
        self.e = e
        return self.get_thetas()

    def residuals(self, data):
        """
        """
        q1, q2 = data
        return ((  q1*self.ab[None,:] \
                 + q2*self.cd[None,:] ).sum(1) + self.e) ** 2
        
    def get_params(self):
        """
        """
        (a,b), (c,d), e = self.ab, self.cd, self.e
        return a,b,c,d,e

    def get_thetas(self):
        """
            Returns t1,t2 in angle degrees
        """
        a,b,c,d,e = self.get_params()
        return coef_to_angle(c, d), coef_to_angle(a, b) 
    
    def get_l1(self, q):
        """
        """
        a,b = self.cd
        e = (q*self.ab[None,:]).sum(1) + self.e
        return a, b, e
    
    def get_l2(self, q):
        """
        """
        a, b = self.ab
        e = (q*self.cd[None,:]).sum(1) + self.e
        return a, b, e
    
def filter_outliers(q1, q2, min_samples=5, residual_threshold=1.):
    """
    """
    nkp = q1.shape[0]
    print(f"{nkp} keypoints matched")
    fund, inliers = ransac((q1,q2), FundamentalMatrix, 
                           min_samples=min_samples, 
                           residual_threshold=residual_threshold)
    q1, q2 = q1[inliers], q2[inliers]
    nkp2 = q1.shape[0]
    print(f"{nkp-nkp2} keypoints filtered by ransac ({residual_threshold}). {nkp2} remaings")
    return q1, q2