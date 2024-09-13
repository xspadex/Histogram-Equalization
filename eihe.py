import numpy as np
import cv2 as cv
from utils import *

def calculate_edge_intensity(vb):
    h, w = vb.shape
    edge_intensity = np.zeros_like(vb, dtype=np.float32)
    
    for y in range(1, h-1):
        for x in range(1, w-1):
            center = int(vb[y, x])
            neighbors = [
                int(vb[y-1, x-1]), int(vb[y-1, x]), int(vb[y-1, x+1]),
                int(vb[y, x-1]), int(vb[y, x+1]),
                int(vb[y+1, x-1]), int(vb[y+1, x]), int(vb[y+1, x+1])
            ]
            edge_intensity[y, x] = sum(abs(center - n) for n in neighbors)
    
    return edge_intensity

def compute_edge_intensity_histogram(edge_intensity, vb):
    L = 256
    E = np.zeros(L, dtype=np.float32)
    
    for m in range(L):
        E[m] = np.sum(edge_intensity[vb == m])
    
    return E

def compute_transformation_function(E, alpha=0.5):
    L = 256
    # 计算CDF
    F = np.cumsum(E)
    F = F / F[-1]  # 归一化

    F_power = F ** alpha
    T = ((L - 1) * F_power).astype(np.uint8)
    
    return T

# 自动判断图像亮度，决定alpha值
def determine_alpha(image, threshold=0.5):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    normalized_hist = hist / hist.sum()
    
    mean_brightness = np.sum(normalized_hist * np.arange(256))

    if mean_brightness < threshold * 255:
        alpha = 0.5 
    else:
        alpha = 1.1  

    return alpha

def edge_intensity_histogram_equalization(image):
    # rgb to hsv
    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(image_hsv)

    # 分成基底层Vd和细节层Vb处理
    lpf_kernel = np.array([[1, 2, 1], 
                       [2, 4, 2], 
                       [1, 2, 1]], dtype=np.float32) / 16.0

    vb = cv.filter2D(v, -1, lpf_kernel)
    vd = cv.subtract(v, vb)

    edge_intensity = calculate_edge_intensity(vb)
    E = compute_edge_intensity_histogram(edge_intensity, vb)


    alpha = determine_alpha(v, 0.5)
    print(alpha)
    T = compute_transformation_function(E, alpha)

    equalized_vb = T[vb]

    epsilon = 1e-8
    equalized_vd = cv.multiply(cv.divide(equalized_vb.astype(np.float32), np.maximum(vb.astype(np.float32), epsilon)), vd.astype(np.float32))

    equalized_vd = np.clip(equalized_vd, 0, 255)
    
    # 计算equalized_v
    equalized_v = cv.add(equalized_vb, equalized_vd.astype(np.uint8))
    equalized_v = equalized_v.astype(np.uint8)

    equalized_image_hsv = cv.merge([h, s, equalized_v])

    # hsv to rgb
    equalized_image_rgb = cv.cvtColor(equalized_image_hsv, cv.COLOR_HSV2BGR)

    # 显示转换后的图像
    # cv.imshow('RGB Image', equalized_image_rgb)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return equalized_image_rgb