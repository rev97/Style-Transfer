from __future__ import print_function

import time
import imageio
from PIL import Image
import numpy as np
import h5py

def imread(path,height,width):
    image = Image.open(path)
    return image.resize((height, width))
 
def expand_dimensions(image):
    return np.expand_dims(np.asarray(image, dtype='float32'), axis=0)


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    imageio.imsave(path, img)

def imgpreprocess(image,mean):
    image = image[np.newaxis,:,:,:]
    return image - mean

def imgunprocess(image,mean):
    temp = image + mean
    return temp[0] 

# function to convert 2D greyscale to 3D RGB
def to_rgb(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret

def adjust_images(content,style):
    content[:,:,:,0] -= 103.99
    content[:, :, :, 1] -= 116.779
    content[:, :, :, 2] -= 123.68
    content = content[:, :, :, ::-1]

    style[:, :, :, 0] -= 103.939
    style[:, :, :, 1] -= 116.779
    style[:, :, :, 2] -= 123.68
    style = style[:, :, :, ::-1]
    return content, style

