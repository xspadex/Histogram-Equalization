import cv2
import numpy as np
from math import *

def Grad_func(img):
    '''
    This Function returns Local Gradient Matrices.
    Parameters
    ----------
    img : Image
          Input Image for which Local Gradient Matrices are to be found.

    Returns
    -------
    Grad_mat : 2D Numpy Matrix
               Matrix containing Local Gradient Matrix for every pixel in input image.
    '''
    dummy_mat = np.array([[0.00 for j in range(2)] for i in range(9)])
    Grad_mat = [[dummy_mat for j in range(img.shape[1])] for i in range(img.shape[0])]
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            Gij = []
            roi = img[i-1:i+2, j-1:j+2]
            gx = cv2.Sobel(roi,cv2.CV_64F, 1, 0, ksize = 1)
            gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize = 1)
            gx = np.reshape(gx,(9, ))
            gy = np.reshape(gy,(9, ))
            Gij.append(gx)
            Gij.append(gy)
            Grad_mat[i][j] = np.transpose(Gij)
    return Grad_mat
    
def steering_kernel(img):
    '''
    This Function Returns Steering Kernel Matrices.
    Parameters
    ----------
    img : Image

    Returns
    -------
    W : 2D Numpy Matrix
        Matrix containing Steering Kernel Weight Matrices for each pixel in input image.
    '''
    Grad_mat = Grad_func(img*255)
    dummy_mat = np.array([[0 for j in range(3)] for i in range(3)])
    W = [[dummy_mat for j in range(img.shape[1])] for i in range(img.shape[0])]
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            u, s, v = np.linalg.svd(Grad_mat[i][j])     
            v2 = v[1]
            if v2[1] == 0:
                theta = pi/2
            else:
                theta = np.arctan(v2[0]/v2[1])
            sigma = (s[0] + 1.0)/(s[1] + 1.0)
            gamma = sqrt(((s[0]*s[1]) + 0.01)/9)
            Rot_mat = np.array([[cos(theta), sin(theta)], [-(sin(theta)), cos(theta)]])
            El_mat = np.array([[sigma, 0], [0, (1/sigma)]])
            C = gamma*(np.dot(np.dot(Rot_mat, El_mat), np.transpose(Rot_mat)))
            coeff = sqrt(np.linalg.det(C))/(2*pi*(5.76))
            W_i = [[0 for q in range(3)] for p in range(3)]
            for n_i in range(i-1, i+2):
                for n_j in range(j-1, j+2):
                    xi = np.array([i, j])
                    xk = np.array([n_i, n_j])
                    xik = xi - xk
                    wik = coeff*(exp(-(np.dot(np.dot(np.transpose(xik), C), xik))/(11.52)))
                    W_i[n_i-i+1][n_j-j+1] = wik
            W[i][j] = W_i
    return W

def Weighted_Guided_Image_Filter(im, p, r, r2, eps, lamda, N):
    '''
    This Function returns the output for Weighted 
    Guided Image Filter applied on Input Image.
    
    Parameters
    ----------
    im : Guidance Image
    
    p : Input Filter Image

    r : Radius of Kernel
    
    r2 : Radius of Local Window centered at a particular pixel

    eps : Regularization parameter
    
    lamda : small constant dependent on dynamic range
        
    N : Number of Pixels in the Input image

    Returns
    -------
    q : Output Image after WGIF application
    '''
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
    mean_I2  = cv2.boxFilter(im, cv2.CV_64F,(r2,r2))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_p2 = cv2.boxFilter(p, cv2.CV_64F, (r2,r2))
    
    corr_I = cv2.boxFilter(im*im, cv2.CV_64F,(r,r))
    corr_I2 = cv2.boxFilter(im*im,cv2.CV_64F,(r2,r2))
    corr_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
    
    var_I = corr_I - mean_I*mean_I
    var_I2 = corr_I2 - mean_I2*mean_I2
    
    PsiI = ((var_I2+lamda)*np.sum(1/(var_I2 + lamda)))/N

    cov_Ip = corr_Ip - mean_I*mean_p
    
    a_psi = cov_Ip/(var_I + eps/PsiI)
    b_psi = mean_p - (a_psi)*mean_I
    mean_ap = cv2.boxFilter(a_psi,cv2.CV_64F,(r2,r2))
    mean_bp = cv2.boxFilter(b_psi,cv2.CV_64F,(r2,r2))
    qp = mean_ap*im + mean_bp
    return qp


def get_WGIF(im):

    minimum = np.min(im)
    maximum = np.max(im)

    L = (maximum-minimum) * 255
    lamda = (0.001*L)**2
    rows, columns = im.shape
    N = rows*columns
    r = 2
    eps = 0.01
    rd = 3

    WGIF = Weighted_Guided_Image_Filter(im, im, r, rd, eps, lamda, N)
    return  WGIF

