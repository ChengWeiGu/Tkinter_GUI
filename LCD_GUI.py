#Ref1: https://www.itread01.com/content/1547705544.html
#pyinstaller使用方式: https://blog.csdn.net/Sagittarius_Warrior/article/details/78457824
#pyinstaller參數說明: https://zwindr.blogspot.com/2016/01/python-pyinstaller.html
import numpy as np
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import pandas as pd
from tkinter import *
import tkinter.messagebox as mb
from PIL import ImageTk, Image
from os.path import join, abspath, basename, dirname, splitext, isfile, isdir
from os import walk, listdir
import cv2
from Model import *


class APP:

    def __init__(self, root, net):
        
        # self.frame = Frame(root) #container
        # self.frame.pack()

        self.dirname = ""
        self.filename = []
        self.photo_ind = 0
        self.total_samples = 0
        self.judge_result = {}
        self.judge_pair_result = []
        self.num_PASS = 0
        self.num_NG = 0
        self.canvas = None
        # obj1: dirname label
        self.dirname_lab = Label(root, text = 'Source Path', bg = 'silver', fg = 'purple', font= ('Arial',14), width = 10)
        self.dirname_lab.place(x = 10, y = 20)
        # obj2: dirname entry
        self.dirname_entry = Entry(root, show = None, font = ('Arial',12), width = 85)
        self.dirname_entry.place(x = 10, y = 50)
        # obj3: load photos by btn
        self.dirname_btn = Button(root, text = 'Load Files', width = 10, height = 2, command = self.folder_select)
        self.dirname_btn.place(x = 810, y = 40)
        # obj4: compute result by btn
        self.calculate_btn = Button(root, text = 'Compute', width = 10, height = 2, command = self.calculate_result)
        self.calculate_btn.place(x = 900, y = 40)
        # obj5: next photo by btn
        self.next_btn = Button(root, text = '>>', width = 5, height = 2, command = self.change_photo_next)
        self.next_btn.place(x = 900, y = 350)
        # self.next_btn.bind("<Right>", self.change_photo_next)
        # obj6: previous photo by btn
        self.previous_btn = Button(root, text = '<<', width = 5, height = 2, command = self.change_photo_previous)
        self.previous_btn.place(x = 60, y =350)
        # self.previous_btn.bind("<Left>", self.change_photo_previous)
        # obj7: show total retult with PASS or NG by label
        self.total_result_var = StringVar()
        self.show_total_result_l = Label(root, textvariable = self.total_result_var, anchor = 'w', fg = 'black', font = ('Times New Roman',20,'bold'), width = 60)
        self.show_total_result_l.place(x = 240, y =700)
        # obj8: show single image result with PASS or NG by label
        self.single_result_var = StringVar()
        self.show_single_result_l = Label(root, textvariable = self.single_result_var , fg = 'black', font = ('Times New Roman',24,'bold'), width = 10)
        self.show_single_result_l.place(x = 400, y =620)

        root.bind("<Left>", self.change_photo_previous)
        root.bind("<Right>", self.change_photo_next)
        root.bind("<Escape>", lambda event: root.quit())
        # list box for all photos
        # self.photo_var = StringVar()
        # self.photo_list_box = Listbox(root,listvariable = self.photo_var, width = 40, height = 20)
        # self.photo_list_box.place(x = 10, y = 80)
        # self.photo_list_box.bind("<Double-Button-1>", self.show_selected_photo)
        # self.photo_list_box.bind("<Up>", self.show_selected_photo)
        # self.photo_list_box.bind("<Down>", self.show_selected_photo)
    #選擇檔案並搜尋圖片路徑
    def folder_select(self):
        try:
            self.dirname = []
            self.dirname = filedialog.askdirectory()
            self.dirname_entry.delete(0,'end')
            self.dirname_entry.insert(0,self.dirname)
            self.set_filename()
        except:
            mb.showerror(title = 'Figure Loading Error', message='Please Select a Folder!')
        if self.total_samples == 0:
            mb.showerror(title = 'Figure Loading Error', message='No BMP Figure exists!')
    #folder select會使用到
    def set_filename(self):
        self.filename = []
        for ind, f in enumerate(listdir(self.dirname)):
            if f.endswith(".bmp"): 
                self.filename += [join(self.dirname, f)]
                # self.photo_list_box.insert(ind,f) #同時show出listbox內容
            else: print("bmp format is required!")
        # self.photo_list_box.activate(0) #listbox的初始設定
        self.total_samples = len(self.filename)
        print(self.total_samples)

    #切換每張圖時顯示PASS or NG
    def set_single_result(self, single_result):
        self.single_result_var.set(single_result) 
        if single_result == "PASS":  self.show_single_result_l.configure(fg = "Blue")
        else: self.show_single_result_l.configure(fg = "Red")

    #計算與預測結果
    def calculate_result(self):
        self.dirname = self.dirname_entry.get()
        if self.dirname != "":
            self.set_filename()
            self.judge_result = net.predict_result(self.filename)
            # print(self.filename)
            self.judge_pair_result = list(self.judge_result.items())
            # print(self.judge_pair_result)
            #初始顯示
            self.show_fig(self.judge_pair_result[self.photo_ind % self.total_samples]) #第一張圖像
            single_result = self.judge_pair_result[self.photo_ind % self.total_samples][1] #第一張圖判定結果
            self.set_single_result(single_result)
            #總結果顯示
            judge_temp = np.array(list(self.judge_result.values()))
            self.num_PASS = np.sum(judge_temp == "PASS")
            self.num_NG = np.sum(judge_temp == "NG")
            self.total_result_var.set("Total samples: {} ea; PASS: {} ea; NG: {} ea".format(self.total_samples,self.num_PASS,self.num_NG))
        else: mb.showerror(title = 'Folder ERROR', message='Please Select Correct Folder!')
    #顯示canvas圖片
    def show_fig(self, pair_result):
        image_path, judge = pair_result[0], pair_result[1]
        plt.close()
        Fig, ax = plt.subplots(1,1, figsize = (7,5))
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis('off'), ax.set_title("{}-{}".format(basename(image_path),judge),fontsize = 18)
        self.canvas = FigureCanvasTkAgg(Fig, master = root)
        self.canvas.get_tk_widget().place(x=150,y=120)
        self.canvas.draw()
    #next button對應函數
    def change_photo_next(self,event=None):
        if self.total_samples != 0:
            self.photo_ind = self.photo_ind + 1
            ind = self.photo_ind % self.total_samples
            self.show_fig(self.judge_pair_result[ind]) #畫下一張圖
            judge_temp = list(self.judge_result.values())
            self.set_single_result(judge_temp[ind]) #下一張圖判定結果

    #previous button對應函數
    def change_photo_previous(self,event=None):
        if self.total_samples != 0:
            self.photo_ind = self.photo_ind - 1
            ind = self.photo_ind % self.total_samples
            self.show_fig(self.judge_pair_result[ind]) #畫上一張圖
            judge_temp = list(self.judge_result.values())
            self.set_single_result(judge_temp[ind])  #上一張圖判定結果
    
    def key(self,event):
        print(event.char)

    #listbox的綁定事件
    # def show_selected_photo(self,event):
    #     Fig, ax = plt.subplots(1,1, figsize = (4,3))
    #     self.selected = self.photo_list_box.get(self.photo_list_box.curselection())
    #     print(self.selected)
    #     img = cv2.imread(join(self.dirname,self.selected))
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     ax.imshow(img)
    #     ax.axis('off'), ax.set_title(self.selected)
    #     convas = FigureCanvasTkAgg(Fig, master = root)
    #     convas.get_tk_widget().place(x=300,y=80)
    #     convas.draw()
        


#create 模型
net = effnet()
#建立視窗
root = Tk()
root.title("LCD Recognition System")
root.geometry("1000x800")
#元件內容
app = APP(root, net)

root.mainloop()

