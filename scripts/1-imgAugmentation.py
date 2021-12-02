# this package tried to rotate, crop, brighten and flip images
# 翻轉方法: https://zhuanlan.zhihu.com/p/45340399
# tf各項影像演 https://zhuanlan.zhihu.com/p/31264834
# the script requires tensorflow=1.12.0 and scipy=1.2.1

from os import walk, listdir, mkdir, makedirs
from os.path import join, basename, dirname, isfile, isdir, splitext
import tensorflow as tf
from scipy import misc
import numpy as np
import cv2
 
#随机旋转图片
def random_rotate_image(image_file ,dst ,num , angle_tol = 30.0 ,channels = 3):
    
    imag_name = basename(image_file)
    imag_name = splitext(imag_name)[0]
    # dst_path = join(dirname(image_file), 'dst')
    # if (isdir(dst_path) == False): mkdir(dst_path) 
    # dst_path = join(dirname(image_file),'dst')
    dst_path = dst # 3/24 added
    with tf.Graph().as_default():
        tf.set_random_seed(666)
        # tf.random.set_seed(666)
        file_contents = tf.read_file(image_file)
        image = tf.image.decode_image(file_contents, channels=channels)
        image_rotate_en_list = []
        def random_rotate_image_func(image):
            #旋转角度范围
            angle = np.random.uniform(low=-angle_tol, high=angle_tol)
            return misc.imrotate(image, angle, 'bicubic')
        for i in range(num):
            image_rotate = tf.py_func(random_rotate_image_func, [image], tf.uint8)
            image_rotate_en_list.append(tf.image.encode_png(image_rotate))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            results = sess.run(image_rotate_en_list)
            for idx,re in enumerate(results):
                with open(dst_path + "\\" + imag_name + "-rotate-" +str(idx)+'.bmp','wb') as f:
                    f.write(re)
 
#随机左右翻转图片
def random_flip_image(image_file, dst ,num, mode = 0, channels = 3):
    
    imag_name = basename(image_file)
    imag_name = splitext(imag_name)[0]
    # dst_path = join(dirname(image_file), 'dst')
    # if (isdir(dst_path) == False): mkdir(dst_path) 
    # dst_path = join(dirname(image_file),'dst')
    dst_path = dst # 3/24 added
    with tf.Graph().as_default():
        tf.set_random_seed(666)
        # tf.random.set_seed(666)
        file_contents = tf.read_file(image_file)
        image = tf.image.decode_image(file_contents, channels=channels)
        image_flip_en_list = []
        if mode == 0: #左右隨機翻轉
            for i in range(num):
                image_flip = tf.image.random_flip_left_right(image)
                image_flip_en_list.append(tf.image.encode_png(image_flip))
        elif mode == 1: #上下隨機翻轉
            for i in range(num):
                image_flip = tf.image.random_flip_up_down(image)
                image_flip_en_list.append(tf.image.encode_png(image_flip))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            results = sess.run(image_flip_en_list)
            for idx,re in enumerate(results):
                if mode == 0: idx = 2*idx
                elif mode == 1: idx = 2*idx+1
                with open(dst_path + "\\" + imag_name +"-flip-" +str(idx)+'.bmp','wb') as f:
                    f.write(re)
 
#随机变化图片亮度
def random_brightness_image(image_file ,dst ,num , delta = 0.1,channels=3 , mode = 'range'):
    
    imag_name = basename(image_file)
    imag_name = splitext(imag_name)[0]
    # dst_path = join(dirname(image_file), 'dst')
    # if (isdir(dst_path) == False): mkdir(dst_path) 
    # dst_path = join(dirname(image_file),'dst')
    dst_path = dst # 3/24 added
    with tf.Graph().as_default():
        tf.set_random_seed(666)
        # tf.random.set_seed(666)
        file_contents = tf.read_file(image_file)
        image = tf.image.decode_image(file_contents, channels=channels)
        image_bright_en_list = []
        for i in range(num):
            if mode == 'range':image_bright = tf.image.random_brightness(image, max_delta = delta)
            elif mode == 'single':image_bright = tf.image.adjust_brightness(image, delta = delta)
            image_bright_en_list.append(tf.image.encode_png(image_bright))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            results = sess.run(image_bright_en_list)
            for idx,re in enumerate(results):
                with open(dst_path + "\\" + imag_name +"-brtn-" +str(idx)+'.bmp','wb') as f:
                    f.write(re)
 
#随机裁剪图片
def random_crop_image(image_file, num,channels = 3):
    
    imag_name = basename(image_file)
    imag_name = splitext(imag_name)[0]
    dst_path = join(dirname(image_file), 'dst')
    if (isdir(dst_path) == False): mkdir(dst_path) 
    dst_path = join(dirname(image_file),'dst')
    with tf.Graph().as_default():
        tf.set_random_seed(666)
        # tf.random.set_seed(666)
        file_contents = tf.read_file(image_file)
        image = tf.image.decode_image(file_contents, channels=channels)
        image_crop_en_list = []
        for i in range(num):
            #裁剪后图片分辨率保持160x160,3通道
            image_crop = tf.random_crop(image, [480, 640, 3])
            image_crop_en_list.append(tf.image.encode_png(image_crop))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            results = sess.run(image_crop_en_list)
            for idx,re in enumerate(results):
                with open(dst_path + "\\" + imag_name + "-crop-" +str(idx)+'.bmp','wb') as f:
                    f.write(re)
 
#随机变化图片亮度
def random_sharpen_image(image_file, num, lower = 0.1, upper = 0.6 , factor = 0.1 ,channels=3):
    
    imag_name = basename(image_file)
    imag_name = splitext(imag_name)[0]
    dst_path = join(dirname(image_file), 'dst')
    if (isdir(dst_path) == False): mkdir(dst_path) 
    dst_path = join(dirname(image_file),'dst')
    with tf.Graph().as_default():
        tf.set_random_seed(666)
        # tf.random.set_seed(666)
        file_contents = tf.read_file(image_file)
        image = tf.image.decode_image(file_contents, channels=channels)
        image_bright_en_list = []
        for i in range(num):
            # image_bright = tf.image.adjust_contrast(image, contrast_factor = factor)
            # image_bright = tf.image.random_contrast(image, lower = lower, upper = upper)
            image_bright_en_list.append(tf.image.encode_png(image_bright))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            results = sess.run(image_bright_en_list)
            for idx,re in enumerate(results):
                with open(dst_path + "\\" + imag_name +"-contrast-" +str(idx)+'.bmp','wb') as f:
                    f.write(re)


    
def main():
    srcs = [r"D:\Side Work Data\LCD photos\Training\Arrangement_train\0522\0-OK",
            r"D:\Side Work Data\LCD photos\Training\Arrangement_train\0522\1-WL",
            r"D:\Side Work Data\LCD photos\Training\Arrangement_train\0522\2-BL",
            r"D:\Side Work Data\LCD photos\Training\Arrangement_train\0522\3-WS",
            r"D:\Side Work Data\LCD photos\Training\Arrangement_train\0522\4-BS"]

    dsts = [r"D:\Side Work Data\LCD photos\Training\Arrangement_train\0522\train1_5cls\0-OK",
            r"D:\Side Work Data\LCD photos\Training\Arrangement_train\0522\train1_5cls\1-WL",
            r"D:\Side Work Data\LCD photos\Training\Arrangement_train\0522\train1_5cls\2-BL",
            r"D:\Side Work Data\LCD photos\Training\Arrangement_train\0522\train1_5cls\3-WS",
            r"D:\Side Work Data\LCD photos\Training\Arrangement_train\0522\train1_5cls\4-BS"]



    channels = 3
    for src, dst in zip(srcs,dsts):
        if not isdir(dst):
            makedirs(dst)
        else:
            print(dst," exists")
        files = listdir(src)
        for f in files:
            imag_path = join(src, f)
            if isdir(imag_path):
                continue
            else:
                #处理图片，进行N次随机处理，并将处理后的图片保存到输入图片相同的路径下
                random_rotate_image(imag_path, dst, 10, angle_tol = 90.0 ,channels = channels)
                random_flip_image(imag_path, dst, 5, mode = 0,  channels = channels) #左右翻轉
                random_flip_image(imag_path, dst, 5, mode = 1,  channels = channels) #上下翻轉
                random_brightness_image(imag_path, dst, 5, delta = 0.05 ,channels = channels, mode = 'range')
                


if __name__ == "__main__":
    main()
   