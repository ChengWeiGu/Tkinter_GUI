# import time
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import cv2
# import os


def zero_pad_photo(imag, top = 160, bottom = 160, left = 0, right = 0): #3/19新增
    return cv2.copyMakeBorder(imag,top,bottom,left,right,cv2.BORDER_CONSTANT,value=(0,0,0))


def resize(imag,w = 160 ,h = 160):
    # return cv2.resize(imag,(w,h), interpolation = cv2.INTER_NEAREST)
    imag = zero_pad_photo(imag) #3/19改 先 padding再縮放成160*160
    return cv2.resize(imag,(w,h), interpolation = cv2.INTER_LINEAR)


def binarize_photo(imag, mode = 0, cut = 100, blocksize = 11):
    if mode == 0 : return cv2.adaptiveThreshold(imag, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blocksize, 0)
    elif mode == 1: return cv2.threshold(imag, cut, 255, cv2.THRESH_BINARY)[1]


def rotate_photo(imag,θ,x_shift,y_shift):
    # to rotate the photo
    θ_rotate = θ*np.pi/180
    M_rotate = np.array([[np.cos(θ_rotate), -np.sin(θ_rotate), x_shift],
                        [np.sin(θ_rotate), np.cos(θ_rotate), y_shift]])

    return cv2.warpAffine(imag, M_rotate, (x_shift,y_shift))


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
    # print("mean = %f" % mean)
    if mean < 5: #3/19 針對純黑色色片的隨機亮點做以下處理 要求 mean < 5
        imag = binarize_photo(imag, mode = 1, cut = 20)
        imag = dilate_photo(imag,1, 5)
        return binarize_photo(imag, mode = 1, cut = 20), mean
    else:
        temp = (imag-mean)*alpha + mean*beta
        return np.uint8(np.clip(temp, 0, 255)), mean

def gamma_transform(imag, gamma):
    table = np.array([((x/255)**gamma)*255 for x in range(256)]).astype('uint8')
    return cv2.LUT(imag, table)


def preprocess_imgs(filename):
    imag_size = 160
    imag_bgr = cv2.imread(filename) # bgr
    imag_gray = cv2.cvtColor(imag_bgr, cv2.COLOR_BGR2GRAY) #rgray scale

    imag_filter = cv2.bilateralFilter(imag_gray, 7,31,31) #3/24 濾雜訊
    imag_c, mean = contrast_up(imag_filter, alpha = 2, beta = 0.8) #3/24 提升對比度

    imag_bgr_small = resize(imag_bgr,imag_size,imag_size) #壓縮圖片 2/5 新增
    imag_c_small = resize(imag_c,imag_size,imag_size) #壓縮圖片 2/6 新增
    imag_c_small = gray2color(imag_c_small,imag_bgr_small) #灰階轉成彩色圖檔
    
    return imag_bgr_small, imag_c_small