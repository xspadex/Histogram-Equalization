import cv2 as cv
import numpy as np
from math import *
import utils
import CLAHE

def Weighted_Guided_Image_Filter(G, X, r, eps, lamda, N):
    mean_G = cv.boxFilter(G, cv.CV_64F, (r, r))
    mean_X = cv.boxFilter(X, cv.CV_64F, (r, r))
    corr_G = cv.boxFilter(G * G, cv.CV_64F, (r, r))
    corr_GX = cv.boxFilter(G * X, cv.CV_64F, (r, r))
    
    var_G = corr_G - mean_G * mean_G

    Gamma = ((var_G + eps) * np.sum(1/(var_G + eps)))/N
    a = (corr_GX - mean_G * mean_X)/(var_G + lamda / Gamma)
    b = mean_X - a * mean_G
    mean_a = cv.boxFilter(a, cv.CV_64F, (r, r))
    mean_b = cv.boxFilter(b, cv.CV_64F, (r, r))
    Z = mean_a * G + mean_b
    return Z


def get_WGIF(img):
    L = (np.max(img) - np.min(img)) * 255
    eps = (0.001 * L) ** 2
    rows, columns = img.shape
    N = rows * columns
    r = 3
    lamda = 0.01
    return  Weighted_Guided_Image_Filter(img, img, r, eps, lamda, N)

def wgif_based_enhance(SI):
    SIL = get_WGIF(SI)

    rows, columns = SIL.shape
    a = 1 - np.average(SIL)
    SILG = np.zeros_like(SIL)
    for i in range(rows):
        for j in range(columns):
            phi = (SIL[i][j] + a) / (1 + a)
            SILG[i][j] = SIL[i][j] ** phi

    SILGf = np.zeros_like(SILG)
    minimum = np.min(SILG)
    maximum = np.max(SILG)
    for i in range(rows):
        for j in range(columns):
            SILGf[i][j] = (SILG[i][j] - minimum) / (maximum - minimum)
    
    SIR = np.zeros_like(SIL)
    for i in range(rows):
        for j in range(columns):
            SIR[i][j] = SI[i][j] / SIL[i][j]

    minimum = np.min(SIR)
    maximum = np.max(SIR)
    avg = np.average(SIR)
    for i in range(rows):
        for j in range(columns):
            SIR[i][j] = (SIR[i][j] - minimum) / (maximum - minimum) * 255
    SIR = np.round(SIR, 0)
    SIR = SIR.astype(dtype=np.uint8)
    SIR = CLAHE.CLAHE(SIR, clip=2.0, tiles=(8, 8))
    SIR = utils.gray_to_intensity(SIR)
    
    SIRH = get_WGIF(SIR)
    SIRH += avg - np.average(SIRH)
    SIE = np.zeros_like(SIRH)
    for i in range(rows):
        for j in range(columns):
            SIE[i][j] = SILGf[i][j] * SIRH[i][j]

    b = np.average(SIE)
    SIEf = np.zeros_like(SIE)
    for i in range(rows):
        for j in range(columns):
            SIEf[i][j] = 1 / (1 + np.exp(-8 * (SIE[i][j] - b)))

    return SIEf