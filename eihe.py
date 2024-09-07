import numpy as np
import cv2 as cv
from utils import *

# 计算边缘强度
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
    
    # 应用幂律变换
    F_power = F ** alpha
    
    # 计算变换函数
    T = ((L - 1) * F_power).astype(np.uint8)
    
    return T

# 自动判断图像亮度，决定alpha值
def determine_alpha(image, threshold=0.5):
    # 计算图像亮度的归一化直方图
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    normalized_hist = hist / hist.sum()
    
    # 计算图像的平均亮度
    mean_brightness = np.sum(normalized_hist * np.arange(256))

    # 决定 alpha 值
    if mean_brightness < threshold * 255:
        # 图像整体偏暗
        alpha = 0.5 
    else:
        # 图像整体偏亮
        alpha = 1.1  

    return alpha

def edge_intensity_histogram_equalization(image):
    # rgb to hsv
    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(image_hsv)

    # 分成基底层Vd和细节层Vb处理
    # 定义3x3低通滤波器核
    lpf_kernel = np.array([[1, 2, 1], 
                       [2, 4, 2], 
                       [1, 2, 1]], dtype=np.float32) / 16.0

    # 应用自定义低通滤波器获得基底层Vb
    vb = cv.filter2D(v, -1, lpf_kernel)

    # 计算细节层 Vd
    vd = cv.subtract(v, vb)

    # 计算边缘强度
    edge_intensity = calculate_edge_intensity(vb)
    
    # 计算边缘强度直方图
    E = compute_edge_intensity_histogram(edge_intensity, vb)

    # 计算变换函数
    alpha = determine_alpha(v, 0.5)
    print(alpha)
    T = compute_transformation_function(E, alpha)

    # 应用变换函数
    equalized_vb = T[vb]

    # 计算 equalized_vd
    # 防止除以零
    epsilon = 1e-8  # 一个非常小的常数
    equalized_vd = cv.multiply(cv.divide(equalized_vb.astype(np.float32), np.maximum(vb.astype(np.float32), epsilon)), vd.astype(np.float32))

    # 确保所有值都在 0 到 255 之间
    equalized_vd = np.clip(equalized_vd, 0, 255)
    
    # 计算equalized_v
    equalized_v = cv.add(equalized_vb, equalized_vd.astype(np.uint8))
    equalized_v = equalized_v.astype(np.uint8)

    equalized_image_hsv = cv.merge([h, s, equalized_v])

    # HSV转换回RGB
    equalized_image_rgb = cv.cvtColor(equalized_image_hsv, cv.COLOR_HSV2BGR)

    # 显示转换后的图像
    # cv.imshow('RGB Image', equalized_image_rgb)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return equalized_image_rgb