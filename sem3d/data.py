from skimage.io import imread
import os

key = lambda x: f"Mekki_1um_3000_20kV_20mm_{x}.jpg"

extensions = {"jpg", "TIF"}
original  = "/home/tristan/workspace/sem/data/Original/"
rectified = "/home/tristan/workspace/sem/data/Rectified/"


def read_img(path, ds):
    """
    """
    if ds=="MEC":
        return imread(path, as_gray=True).T
    else:
        return (imread(path, as_gray=True)*255).astype("uint8")

def get_keys(ds):
    """
    """
    return list_imgs(os.path.join(original + ds))

def list_imgs(folder):
    return list(filter(lambda x:x.split(".")[-1] in extensions,
                       os.listdir(folder)))
    
def imgs_path(ds):
    """
    """
    return {f: os.path.join(original, ds, f) for f in get_keys(ds)}
        
def get_imgs(ds="MEC"):
    """
    """
    return {k:read_img(v, ds) for k,v in imgs_path(ds).items()}

def get_rectified(ds="MEC", k1=None, k2=None):
    """
    """
    key_pairs = rectified_keys(ds)
    if (k1, k2) in key_pairs:
        im1, im2 = rectified_imgs(ds, k1, k2)
    elif (k2, k1) in key_pairs:
        im1, im2 = rectified_imgs(ds, k2, k1)
    else:
        raise Exception(f"Keys ({k1},{k2}) not in {ds} rectified folder")
    if ds == "MEC":
        return im1.T, im2.T
    else:
        return im1, im2
        
def rectified_imgs(ds, k1, k2):
    """
    """
    return (read_img(f"{rectified}/{ds}/{k1}_{k2}_left.jpg", ds), 
            read_img(f"{rectified}/{ds}/{k1}_{k2}_right.jpg", ds))

def rectified_keys(ds):
    """
    """
    return list(map(lambda x:tuple(x.split("_")[:2]), os.listdir(rectified + ds)))
    
def get_pairs(keys):
    """
    """
    keys = list(keys)
    pairs = []
    for i in range(len(keys)):
        for j in range(i):
            pairs.append((keys[i], keys[j]))
    return pairs