# this program is aimed at saving image data as pkl file for PCB, connector and LCD cases
# note that PCB is gray .jpg; connector is RGB .jpg; LCD is RGB .bmp
# two-class classification considered

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from os.path import basename, join, dirname, isfile, isdir,splitext
from os import listdir, walk
import cv2
import pickle
import tensorflow as tf
from scipy import misc


def random_rotate_image(image_file ,dst ,num , angle_tol = 30.0 ,channels = 3):
    
    imag_name = basename(image_file)
    imag_name = splitext(imag_name)[0]
    dst_path = dst
    with tf.Graph().as_default():
        tf.set_random_seed(666)
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
        

def get_rotate_img(imag, angle, scale = 1.0):
    h, w, c = imag.shape
    center = (w/2, h/2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(imag, matrix, (h, w))


def get_flip_img(imag, mode = 1):
    if mode == 0: return cv2.flip(imag, 0) #vertically
    elif mode == 1 : return cv2.flip(imag, 1) #horizontally


def transform_all_img(image_file ,dst):
    imag = cv2.imread(image_file)
    # imag = cv2.cvtColor(imag, cv2.COLOR_BGR2RGB)
    angles = [90,180,270]
    rot_imgs = [imag] + [get_rotate_img(imag, ang) for ang in angles]
    flip_imgs = [get_flip_img(im) for im in rot_imgs]
    tot_imgs = rot_imgs + flip_imgs
    out_filename_temp = join(dst,basename(image_file))
    for ind, im in enumerate(tot_imgs):
        out_filename = splitext(out_filename_temp)[0] + '-new-{}.bmp'.format(ind)
        cv2.imwrite(out_filename,im)


class lcd_Img2pkl:

    def __init__(self, root, img_size):
        self.root = root
        self.CLASS_LIST = []
        self.label = []
        self.img_arr = []
        self.img_size = img_size

    def SET_CLASS_LIST(self):
        
        for dir_basename in listdir(self.root):
            dir_fullname = join(self.root,dir_basename)
            if isdir(dir_fullname):
                self.CLASS_LIST += [dir_fullname]
        print('the list of class: ',self.CLASS_LIST)

    def READ_IMG(self):
        
        for class_name in self.CLASS_LIST:
            init = 0
            print("current class: {}".format(basename(class_name)))
            for img_basename in listdir(class_name):
                img_fullname = join(class_name,img_basename)
                data = cv2.imread(img_fullname)
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                if basename(class_name).startswith('0'): self.label += [0]
                elif basename(class_name).startswith('1'): self.label += [1]
                elif basename(class_name).startswith('2'): self.label += [2]
                # elif basename(class_name).startswith('3'): self.label += [3]
                # elif basename(class_name).startswith('4'): self.label += [4]
                self.img_arr += [data]
                init += 1
                # if init % 100 == 0 : print(init)
            print("current class: {} finished".format(basename(class_name)))
            
        
        self.img_arr = np.array(self.img_arr).reshape(-1,self.img_size,self.img_size,3)
        self.label = np.array(self.label)
        self.img_arr, self.label = shuffle(self.img_arr, self.label)
        print("the labels = {}".format(self.label))
        # print("the first 2 imgs = {}".format(self.img_arr[:2,:,:,:]))


    def SPLIT_DATA(self, test_size):
        X_train, X_test, y_train, y_test = train_test_split(self.img_arr,self.label, test_size = test_size)
        results = [(X_train,y_train),(X_test,y_test)]
        out_filename = join(self.root,'dataset_train_lcd.pkl')
        with open(out_filename,"wb") as the_file:
            pickle.dump(results, the_file, protocol = 4)
            the_file.close()
            
    def PICKLE_ALL_DATA(self): 
        results = [self.img_arr, self.label]
        out_filename = join(self.root,'dataset_test2_lcd.pkl')
        with open(out_filename,"wb") as the_file:
            pickle.dump(results, the_file, protocol = 4)
            the_file.close()



def main():
    root1 = r"D:\Side Work Data\LCD photos\Training\Arrangement_train\0522\test2_3cls"
    img_size1 = 224
    tool1 = lcd_Img2pkl(root1,img_size1)
    tool1.SET_CLASS_LIST()
    tool1.READ_IMG()
    tool1.PICKLE_ALL_DATA()
    # plt.imshow(tool1.img_arr[4000,:,:,:])
    # print(tool1.label[4000])
    # plt.axis('off')
    # plt.show()


if __name__ == "__main__":
    main()