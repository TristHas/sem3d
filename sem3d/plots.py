import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from .ransac_skim import FundamentalMatrix

def draw_line(a=1, b=1, e=[0], width=800, ax=None, c="red"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,10))

    x = np.array([0,width])
    for e_ in e:
        ax.plot(x, -(a*x + e_) / b, '-', c=c)

def plot_epipolar_lines(img1, img2, q1, q2, ax=None, npoints=5):
    """
    """
    if isinstance(npoints, int):
        idx = np.random.choice(q1.shape[0], size=npoints, replace=False)
    else:
        idx = npoints
    if ax is None:
        fig, ax = plt.subplots(1,2, figsize=(20,10))
    fund = FundamentalMatrix()
    t1,t2 = fund.estimate((q1, q2))
    a1, b1, es1 = fund.get_l1(q1)
    a2, b2, es2 = fund.get_l2(q2)

    ax[0].imshow(img1)
    ax[0].scatter(q1[idx,0], q1[idx,1], c="yellow")
    draw_line(a=a2, b=b2, e=es2[idx], width=img1.shape[1], ax=ax[0], c="red")

    ax[1].imshow(img2)
    ax[1].scatter(q2[idx,0], q2[idx,1], c="yellow")
    draw_line(a=a1, b=b1, e=es1[idx], width=img2.shape[1], ax=ax[1], c="red")
    return t1, t2

def plot_disp(q1, q2, bins=100, ax=None):
    """
    """
    if ax is not None:
        assert len(ax)==3
    else:
        fig, ax = plt.subplots(1,3, figsize=(30,10))
    fund = FundamentalMatrix() 
    fund.estimate((q1, q2))
    res = fund.residuals((q1, q2))
    
    ax[0].hist((q1-q2)[:,0], bins=100)
    ax[1].hist((q1-q2)[:,1], bins=100)
    ax[2].hist(res, bins=100)
    
def plot_disp_trends(q1, q2, ax=None):
    if ax is None:
        fig,ax = plt.subplots(2,2, figsize=(20,20))
    x_y_axis = (q1[:,0], q1[:,1])
    x_y_diff = ((q1-q2)[:,0], (q1-q2)[:,1])

    for i,axis in enumerate(x_y_axis):
        for j,diff in enumerate(x_y_diff):
            ax[j,i].scatter(axis, diff)
    
def plot_patch(img, corner=(200, 400), size=(400, 50), ax=None):
    x,y = corner
    w,h = size
    if ax is None:
        fig,ax = plt.subplots(figsize=(10,10))
    ax.imshow(img)
    ax.add_patch(patches.Rectangle(corner, *size, linewidth=1, edgecolor='r', facecolor='none'))