import numpy as np
from utils import *

def DSIHE(img):
    height, width = img.shape
    print(img.shape)
    img_median = int(np.median(img))
    img_lower = np.zeros(256)
    img_upper = np.zeros(256)
    lower_count = 0
    upper_count = 0
    
    for i in range(height):
        for j in range(width):
            if(img[i,j] > img_median):
                upper_count += 1
                img_upper[img[i,j]] += 1
            else:
                lower_count += 1
                img_lower[img[i,j]] += 1
    
    lower_p = img_lower / lower_count
    upper_p = img_upper / upper_count
    lower_cul_p = np.cumsum(lower_p)
    upper_cul_p = np.cumsum(upper_p)
    #print("lower_cul_p: ",lower_cul_p)
    #print("upper_cul_p: ",upper_cul_p)

    #former gray to equalized gray
    lower_min = np.min(img)
    lower_max = img_median
    upper_max = np.max(img) 
    upper_min = img_median + 1 
    lower_gray = np.zeros(256)
    upper_gray = np.zeros(256)

    #print("lower_min: ",lower_min)
    #print("lower_max: ",lower_max)
    #print("upper_min: ",upper_min)
    #print("upper_max: ",upper_max)
    
    for i in range(0,img_median+1):
        lower_gray[i] = (lower_min + (lower_max-lower_min) * lower_cul_p[i]).astype(np.uint8)
    for i in range(img_median+1,256):
        upper_gray[i] = (upper_min + (upper_max-upper_min) * upper_cul_p[i]).astype(np.uint8)
    
    #print("lower_gray: ",lower_gray)
    #print("upper_gray: ",upper_gray)
    DSIHE_img = np.copy(img)
    for i in range(height):
        for j in range(width):
            if(img[i,j] > img_median):
                DSIHE_img[i,j] = upper_gray[img[i,j]]
            else:
                DSIHE_img[i,j] = lower_gray[img[i,j]]
    return DSIHE_img