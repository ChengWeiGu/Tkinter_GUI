import numpy as np
import cv2
from os.path import join, basename, dirname, splitext, isfile, isdir
from os import walk, listdir
import random
import argparse
from copy import deepcopy
import pickle


def load_data(dataset_src):
    with open(dataset_src, "rb") as the_file:
        return pickle.load(the_file)
        the_file.close()

def save_data(dataset_dst,data):
    with open(dataset_dst,"wb") as the_file:
        pickle.dump(data,the_file, protocol = 4)
        the_file.close()


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


def binarize_photo(imag, mode = 0, cut = 100, blocksize = 11):
    if mode == 0 : return cv2.adaptiveThreshold(imag, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, 0) # ADAPTIVE_THRESH_MEAN_C
    elif mode == 1: return cv2.threshold(imag, cut, 255, cv2.THRESH_BINARY)[1]

def contrast_up(imag, alpha = 0.2, beta = 0.8):
    mean = np.average(imag)
    # print("mean = %f" % mean)
    if mean < 5: #4/24 特別針對純黑色色片的隨機白色亮點做以下處理，要求 mean < 5
        imag = binarize_photo(imag, mode = 1, cut = 20)
        imag = dilate_photo(imag,1, 5)
        return binarize_photo(imag, mode = 1, cut = 20), mean
    else: #其它色片都當作白色色片處理 
        temp = deepcopy(imag)
        temp_film = temp[temp > 8]
        mean_film = temp_film.mean()
        # print("test = ",mean_film)
        temp = (temp-mean_film)*alpha + temp*beta
        # temp = (temp-mean_film)*alpha + mean_film*beta
        temp[imag < 8] = 0
        return np.uint8(np.clip(temp, 0, 255)), mean

# 4/24 新增，show grayscale distribution
def show_hist(imag_gray, lower_limit = 0, upper_limit = 255, label = None):
    hist = cv2.calcHist([imag_gray],[0],None,[256],[0,256])
    plt.bar(np.arange(lower_limit+1 , upper_limit+1, 1),hist.ravel()[lower_limit:upper_limit], label = label)
    plt.title(None)
    plt.show()

# 4/24 新增，calc the image background
def calc_img_by_bg(imag_gray, cut = 30):
    img_temp = deepcopy(imag_gray)
    img_temp[img_temp < cut] = 0
    # erode_photo(img_temp,1,5)
    return img_temp


def zero_pad_photo(imag, top = 912, bottom = 912, left = 0, right = 0): #3/19新增
    return cv2.copyMakeBorder(imag,top,bottom,left,right,cv2.BORDER_CONSTANT,value=0)


def resize(imag,w = 160 ,h = 160, args = (912,912,0,0)):
    # return cv2.resize(imag,(w,h), interpolation = cv2.INTER_NEAREST)
    imag = zero_pad_photo(imag, *args) #3/19改 先 padding再縮放成160*160
    return cv2.resize(imag,(w,h), interpolation = cv2.INTER_LINEAR)

def gray2color(imag_gray,imag_bgr):
    img2 = np.zeros_like(imag_bgr)
    img2[:,:,0] = imag_gray
    img2[:,:,1] = imag_gray
    img2[:,:,2] = imag_gray
    return img2

def preprocess_image(filename):
    imag_size = 224
    img_bgr = cv2.imread(filename) # bgr
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray[img_gray < 8] = 0

    imag_filter = cv2.bilateralFilter(img_gray, 5,22,22)
    
    imag_c, mean = contrast_up(imag_filter, 0.2, 0.8) #2/6 提升對比度

    img_bgr_small = resize(img_bgr,imag_size,imag_size) #壓縮BGR圖片

    imag_c_erode = erode_photo(imag_c,1, 31)
    imag_c_dilate = dilate_photo(imag_c,1, 31)
    # imag_c_resize = resize(imag_c, w = 160, h = 160,args = (912,912,0,0))
    imag_c_resize_e = resize(imag_c_erode, w = imag_size, h = imag_size,args = (912,912,0,0))
    imag_c_resize_d = resize(imag_c_dilate, w = imag_size, h = imag_size,args = (912,912,0,0))

    return gray2color(imag_c_resize_e, img_bgr_small), gray2color(imag_c_resize_d, img_bgr_small)


    
def main():
    src = "../BU_Raw"
    for cls_dir in listdir(src):
        cls_dirname = join(src,cls_dir)
        for img_basename in listdir(cls_dirname):
            img_filename = join(cls_dirname,img_basename)
            if isdir(img_filename):
                continue
            else:
                img_erode, img_dilate = preprocess_image(img_filename)
                img_erode_filename = cls_dirname + '/erode/' + splitext(img_basename)[0] + '-erode.png'
                img_dilate_filename = cls_dirname + '/dilate/' + splitext(img_basename)[0] + '-dilate.png'
                cv2.imwrite(img_erode_filename,img_erode)
                cv2.imwrite(img_dilate_filename,img_dilate)
                print(img_basename, ' finished!')

#從1280*960的160*160轉換大小
def main_from_previous():
    src = r"D:\Side Work Data\LCD photos\3rd photo test\New Classes\All Test\test11\0-OK-New"
    dst = r"D:\Side Work Data\LCD photos\Training\Arrangement_train\0522\0-OK"
    imag_size = 224
    for img_basename in listdir(src):
        img_filename_src = join(src, img_basename)
        img_bgr_src = cv2.imread(img_filename_src)
        img_bgr_dst = cv2.resize(img_bgr_src,(imag_size,imag_size), interpolation = cv2.INTER_LINEAR)
        img_filename_dst = join(dst, img_basename)
        cv2.imwrite(img_filename_dst,img_bgr_dst)
        print(img_basename, ' finished!')


if __name__ == "__main__":
    # main()
    main_from_previous()
   