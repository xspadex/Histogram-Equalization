import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img = cv.imread("./pics/sample01.jpg", cv.IMREAD_GRAYSCALE)
print(img)

def draw_histogram(hist, bins):
    plt.figure()
    plt.title("Histogram of Pixel Intensities")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])  # 设置x轴的范围
    plt.bar(bins[:-1], hist, width=1, align='edge')  # 绘制直方图
    plt.show()

def manual_equalize_hist(image):
    # 计算直方图
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # 计算累积直方图
    norm_hist = hist / hist.sum()
    acc_hist = np.cumsum(norm_hist)

    draw_histogram(norm_hist, bins)

    # 找到最小非零累积概率
    min_val = np.min(acc_hist[acc_hist > 0])

    # 计算转换映射
    sk_map = np.round(255 * acc_hist)
    # cdf = np.uint8(255 * (acc_hist - min_val) / (1 - min_val))

    # 应用转换函数
    equalized_image = sk_map[image]

    new_hist, new_bins = np.histogram(equalized_image.flatten(), 256, [0, 256])

    draw_histogram(new_hist, new_bins)

    return equalized_image

hed = manual_equalize_hist(img)




# plt.figure()
# plt.title("Histogram of Pixel Intensities")
# plt.xlabel("Pixel Intensity")
# plt.ylabel("Frequency")
# plt.xlim([0, 256])  # 设置x轴的范围
# plt.bar(bin_edges[:-1], hist, width=1, align='edge')  # 绘制直方图
# plt.show()

# cv.imshow("original",img)
# cv.waitKey(0)
# cv.destroyAllWindows()
# cv.imshow("hed",hed)
# cv.waitKey(0)
# cv.destroyAllWindows()