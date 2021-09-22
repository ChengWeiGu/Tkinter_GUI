# import time
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import cv2
# import os


def resize(imag,w = 160 ,h = 160):
    # return cv2.resize(imag,(w,h), interpolation = cv2.INTER_NEAREST)
    return cv2.resize(imag,(w,h), interpolation = cv2.INTER_LINEAR)
    # print(imag_200x200.shape)

def erode_photo(imag, iter=1, k_size = 15, mag = 1):
    # define the kernel
    kernel = np.ones((k_size,k_size))*mag
    # erode
    return cv2.erode(imag, kernel, iterations = iter)

def dilate_photo(imag, iter=1, k_size = 15, mag = 1):
    # define the kernel
    kernel = np.ones((k_size,k_size))*mag
    # dilate
    return cv2.dilate(imag, kernel, iterations = iter)

def gray2color(imag_gray,imag_bgr):
    img2 = np.zeros_like(imag_bgr)
    img2[:,:,0] = imag_gray
    img2[:,:,1] = imag_gray
    img2[:,:,2] = imag_gray
    return img2

def contrast_up(imag, alpha = 0.2, beta = 0.9):
    mean = np.average(imag)
    temp = (imag-mean)*alpha + mean*beta
    return np.uint8(np.clip(temp, 0, 255))

def preprocess_imgs(filename):
    imag_size = 160
    imag_bgr = cv2.imread(filename) # bgr
    imag_gray = cv2.cvtColor(imag_bgr, cv2.COLOR_BGR2GRAY) #rgray scale
    imag_filter = cv2.bilateralFilter(imag_gray, 7,31,31) #2/6 濾雜訊
    imag_c = contrast_up(imag_filter, alpha = 2, beta = 0.8) #2/6 提升對比度
    imag_bgr_small = resize(imag_bgr,imag_size,imag_size) #壓縮圖片 2/5 新增
    imag_c_small = resize(imag_c,imag_size,imag_size) #壓縮圖片 2/6 新增

    imag_erode = erode_photo(imag_c,3, 5)
    imag_erode = resize(imag_erode,imag_size,imag_size) #壓縮圖片 2/5 新增
    imag_erode = gray2color(imag_erode, imag_bgr_small) #灰階存成彩色圖檔

    imag_dilate = dilate_photo(imag_c,5, 5)
    imag_dilate = resize(imag_dilate,imag_size,imag_size) #壓縮圖片 2/5 新增
    imag_dilate = gray2color(imag_dilate, imag_bgr_small) #灰階存成彩色圖檔

    return imag_bgr_small , imag_erode, imag_dilate