import cv2
import numpy as np

def bi_histogram_equalization(image, lower_limit=1000, upper_limit=2000):
    # 确保图像为灰度图像
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 计算图像的直方图
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    print('------------test for image--------------')
    print(hist)
    
    # 将直方图分为低灰度和高灰度两个部分
    mid_point = np.median(image)
    lower_hist = hist[:int(mid_point)]
    upper_hist = hist[int(mid_point):]
    print(lower_hist)
    print(upper_hist)

    # 对低灰度部分和高灰度部分分别进行剪切处理
    lower_hist = np.clip(lower_hist, 0, lower_limit)
    upper_hist = np.clip(upper_hist, 0, upper_limit)

    # 分别计算累积分布函数 (CDF)
    lower_cdf = lower_hist.cumsum()
    upper_cdf = upper_hist.cumsum()
    # 归一化 CDF
    lower_cdf = (lower_cdf - lower_cdf.min()) / (lower_cdf.max() - lower_cdf.min()) * (mid_point - 1)
    upper_cdf = (upper_cdf - upper_cdf.min()) / (upper_cdf.max() - upper_cdf.min()) * (255 - mid_point) + mid_point

    # 合并两个部分的 CDF
    full_cdf = np.concatenate((lower_cdf, upper_cdf))

    # 根据 CDF 映射原始图像的像素值
    equalized_image = np.interp(image.flatten(), bins[:-1], full_cdf)
    equalized_image = equalized_image.reshape(image.shape).astype('uint8')

    return equalized_image
