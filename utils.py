import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def read_img_gray(path):
    return cv.imread(path, cv.IMREAD_GRAYSCALE)


def draw_matrix(matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(matrix.shape[1]), np.arange(matrix.shape[0]))
    ax.plot_surface(X, Y, matrix, cmap='viridis')
    plt.show()


def draw_2_matrixs(matrixs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(matrixs[0].shape[1]), np.arange(matrixs[0].shape[0]))
    ax.plot_surface(X, Y, matrixs[0], cmap='viridis')

    fig = plt.figure()
    bx = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(matrixs[1].shape[1]), np.arange(matrixs[1].shape[0]))
    bx.plot_surface(X, Y, matrixs[1], cmap='viridis')
    plt.show()


def draw_2_matrixs_in_one(matrixs):
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(121, projection='3d')
    X, Y = np.meshgrid(np.arange(matrixs[0].shape[1]), np.arange(matrixs[0].shape[0]))
    ax.plot_surface(X, Y, matrixs[0], cmap='viridis')
    ax.set_title("figure1")

    bx = fig.add_subplot(122, projection='3d')
    X, Y = np.meshgrid(np.arange(matrixs[1].shape[1]), np.arange(matrixs[1].shape[0]))
    bx.plot_surface(X, Y, matrixs[1], cmap='viridis')
    bx.set_title("figure2")
    plt.show()


def show_image(image, title="default"):
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def show_images_concat(img1, img2):
    n_rows = max(img1.shape[0], img2.shape[0])
    canvas = np.zeros((n_rows, img1.shape[1] + img2.shape[1]), dtype=np.uint8)

    # 将img1和img2绘制到画布上
    canvas[:img1.shape[0], :img1.shape[1]] = img1
    canvas[:img2.shape[0], img1.shape[1]:] = img2

    # 显示画布
    cv.imshow('Concat Image', canvas)
    cv.waitKey(0)
    cv.destroyAllWindows()


def draw_histogram(hist, bins):
    plt.figure()
    plt.title("Histogram of Pixel Intensities")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])  # 设置x轴的范围
    plt.bar(bins[:-1], hist, width=1, align='edge')  # 绘制直方图
    plt.show()


def draw_histograms_in_one(hist1, bins1, hist2, bins2):
    plt.figure(figsize=(12, 6))
    plt.title("Histogram of Pixel Intensities")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])  # 设置x轴的范围
    plt.bar(bins1[:-1], hist1, width=1, align='edge')  # 绘制直方图1
    plt.bar(bins2[:-1], hist2, width=1, align='edge')  # 绘制直方图2
    plt.show()
