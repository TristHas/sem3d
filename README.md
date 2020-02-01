# Repo for the project of 3D SEM reconstruction

## 1. To do

### Finish thesis replication
- Fix the y-shift (probably intrinsic parameter estimation?)
- Add triangulation code
- Propper mesh plotting

### Multi-view integration
- How do combine multiple views?

### Integration of depth from shading and stereo
- How to integrate the luminance properties?

### Gradually introduce learning
- Sparse reconstruction
 - Kornia, Affnet, etc.
 
- Dense reconstruction
 - CNN-dot product, diff DP. 
 
## 2. Ideas

### Hand-crafted features, Learning and differentiable rendering


## 3. Organisation

There are many things to study to actually make it work.
This section tries to organize the different topics that need to be mastered to have a reliable solutions.

### Geometry

The theory behind sparse keypoint matching seems to be geometrical.
In particular, we want feature descriptors invariant or co-variant to affine transformations, due to the orthographic projection model we use.
We should fully understand the theory behind handcrafted features, the state of learned vs. handcrafted feature extraction, etc.
A good starting point is: Repeatibility is not enough: Learning Affine Region via Discriminability.

### Illumination

Another big topic is the study of the pixel intensity.
This will go through the study of rendering + Electron microscopy.
A good starting point is Differentiable Visual Computing Thesis.

### Validation

We do not have 3D groundtruth.
Hence, we need to think about ways to validate our reconstructions. 

- How to measure the accuracy? (TODO) - We can use the delta y after rectification - However, outliers are filtered based on this delta_y If so, we need some very good feature points. Manual selection of ground truth?


