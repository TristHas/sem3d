# Repo for the project of 3D SEM reconstruction

## Already Done
### 1. Fix current code:
- Centering does not work for some negative values (Done) - Validate roatation on artificial rotations (Done) - fundamental_matrix does not work on non-centered images (Done)
### 2. Robust regression
- Adapt skimage ransac to our regressor (Done) - Other algos work better (cf thesis paper) than RANSAC. Try these algorithms instead (TODO). - Then try and compare the pytorch robust loss (TODO).
### Improve visualization:
- Draw epipolar lines (OK) - Draw rectangles (OK)
### Results validation
- How to measure the accuracy? (TODO) - We can use the delta y after rectification - However, outliers are filtered based on this delta_y If so, we need some very good feature points. Manual selection of ground truth?
### Dense matching
- Try out some matching on small image patches. (TODO) - Investigate effect of patch size and feature descriptor (census vs. SAD vs. correlation) (TODO) - Calibrate the P1 and P2 of sgm - Is deep learning bringing something for dense matching
## Todo
### Sparse reconstruction features
- Improve on RANSAC using better regressors - Autocalibration of intrinsic camera parameters
### Sparse reconstruction reorganization
- Clean notebook to plot errors at each step
### Dense reconstruction
- Try out some matching on small image patches. - Investigate effect of patch size and feature descriptor (census vs. SAD vs. correlation) - Calibrate the P1 and P2 of sgm
- Is deep learning bringing something for dense matching
