import cv2
import numpy as np
from math import *

def dph_equalization(image, low_clip=0.01, high_clip=0.9):
    
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    hist, bins = np.histogram(image.flatten(), 256, [0,256])
    
    hist = hist.astype('float32') / hist.sum()
    
    low_clip_value = np.clip(low_clip * hist.sum(), 0, 1)
    high_clip_value = np.clip(high_clip * hist.sum(), 0, 1)
    
    low_clip_indices = hist < low_clip_value
    low_clip_sum = hist[low_clip_indices].sum()
    hist[low_clip_indices] = 0
    
    high_clip_indices = hist > high_clip_value
    high_clip_sum = hist[high_clip_indices].sum()
    hist[high_clip_indices] = high_clip_value
    
    hist += (low_clip_sum + high_clip_sum) / hist.size
    
    cdf = hist.cumsum()
    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min()) * 255
    
    equalized_image = np.interp(image.flatten(), bins[:-1], cdf)
    equalized_image = equalized_image.reshape(image.shape).astype('uint8')
    
    return equalized_image