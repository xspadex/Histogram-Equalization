from wgif import get_WGIF
import cv2
import numpy as np


pic_list = [
    "sample01.jpg",
    "sample02.jpeg",
    "sample03.jpeg",
    "sample04.jpeg",
    "sample05.jpeg",
    "sample06.jpg",
    "sample07.jpg",
    "sample08.jpg",
]

for pic in pic_list:

    img = cv2.imread("./pics/" + pic, 1)
    bgr = np.float64(img)

    SI = np.divide(bgr[:,:,0] + bgr[:,:,1] + bgr[:,:,2] + 0.001, 3 * 255)
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

    SIRH = get_WGIF(SIR)
    SIE = np.zeros_like(SIRH)
    for i in range(rows):
        for j in range(columns):
            SIE[i][j] = SILGf[i][j] * SIRH[i][j]

    b = np.average(SIE)
    SIEf = np.zeros_like(SIE)
    for i in range(rows):
        for j in range(columns):
            SIEf[i][j] = 1 / (1 + np.exp(-8 * (SIE[i][j] - b)))

    for i in range(rows):
        for j in range(columns):
            for k in range(3):
                bgr[i][j][k] = bgr[i][j][k] * SIEf[i][j] / SI[i][j]

    cv2.imwrite("results/" + pic.replace(".jpeg", ".jpg"), bgr)