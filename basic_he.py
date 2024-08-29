import numpy as np
from utils import *


def get_histogram(image):
    return np.histogram(image.flatten(), 256, (0, 256))


def histogram_equalize(image):
    # 计算直方图
    hist, bins = get_histogram(image)
    # 计算归一化累积直方图
    norm_hist = hist / hist.sum()
    acc_hist = np.cumsum(norm_hist)

    # 计算转换映射
    sk_map = (255 * acc_hist).astype(np.uint8)

    # 应用转换函数
    equalized_image = sk_map[image]

    return equalized_image