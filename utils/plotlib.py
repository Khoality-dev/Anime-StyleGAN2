import matplotlib.pyplot as plt 
import cv2
import numpy as np
import tkinter as tk

def plot_multiple_vectors(v, figsize = (15,5), title = None, xlabel = None, ylabel = None, legends = None, save_path = None, show = False):
    plt.figure(figsize = figsize)
    for vector in v:
        plt.plot(vector)
    if title!=None:
        plt.title(title)
    if xlabel != None:
        plt.xlabel(xlabel)
    if ylabel != None:
        plt.ylabel(ylabel)
    if legends!= None:
        plt.legend(legends)
    if save_path != None:
        plt.savefig(save_path)
    if show == True:
        plt.show()
    else:
        plt.close()

def process_display_image(img_list):
    n = len(img_list)
    img_list = list(np.array(img_list).astype(np.uint8))
    size = int(np.sqrt(n))
    cnt = 0
    out_image = None
    for i in range(size):
        temp = img_list[cnt]
        cnt += 1
        for _ in range(size-1):   
            temp = np.hstack((temp,img_list[cnt]))
            cnt += 1
        if i==0:
            out_image = temp.copy()
        else:
            out_image = np.vstack((out_image, temp.copy()))
    return out_image

def display_img(img_list, save_path = None, show = False):
    
    image = process_display_image(img_list)
    root = tk.Tk()
    resolution = min(root.winfo_screenwidth(), root.winfo_screenheight())
    header_size = 100
    image = cv2.resize(image, (resolution - header_size, resolution - header_size))

    if show:
        cv2.imshow('Preview', image)
        cv2.waitKey(0)
    
    if save_path!=None:
        cv2.imwrite(save_path, image)