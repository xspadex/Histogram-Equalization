import numpy as np
from utils import *


def histogram_equalize(image, need_draw_histogram = False):
    # 计算直方图
    hist, bins = np.histogram(image.flatten(), 256, (0, 256))

    # 计算归一化累积直方图
    norm_hist = hist / hist.sum()
    acc_hist = np.cumsum(norm_hist)

    # 计算转换映射
    sk_map = (255 * acc_hist).astype(np.uint8)

    # 应用转换函数
    equalized_image = sk_map[image]

    if need_draw_histogram:
        new_hist, new_bins = np.histogram(equalized_image.flatten(), 256, (0, 256))
        draw_histograms_in_one(hist, bins, new_hist, new_bins)

    return equalized_image