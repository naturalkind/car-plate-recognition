#-*- coding: utf-8 -*-
#
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob, random
import json

def imgs(x):
      cv2.imshow('Rotat', np.array(x))
      cv2.waitKey(0)
      cv2.destroyAllWindows()

def from_yolo_to_cor(box, shape):
    img_h, img_w, _ = shape
    #------------
    x1, y1 = int((box[0] + box[2]/2)*img_w), int((box[1] + box[3]/2)*img_h)
    x2, y2 = int((box[0] - box[2]/2)*img_w), int((box[1] - box[3]/2)*img_h)
    #WORK
    #x1, y1 = int((box[0] + box[2]/2)*416), int((box[1] + box[3]/2)*416)
    #x2, y2 = int((box[0] - box[2]/2)*416), int((box[1] - box[3]/2)*416)
    return x1, y1, x2, y2

def draw_boxes(img, boxes):
    #for box in boxes:
    x1, y1, x2, y2 = from_yolo_to_cor(boxes, img.shape)
    print (x1, y1, x2, y2)
    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 3)
    

class DATA(object):
        def __init__(self):
            self.file = {}
            
        def parseIMG(self, dir_name):
                path = dir_name+"/"
                print ("PARSING",path)
                for r, d, f in os.walk(path):
                    for ix, file in enumerate(f):
                        #print (file)
                        if ".jpg" in file:
                           self.file[file.split(".")[0]] = [os.path.join(r, file)]
                        if ".jpeg" in file:
                           self.file[file.split(".")[0]] = [os.path.join(r, file)]
                        if ".png" in file:
                           self.file[file.split(".")[0]] = [os.path.join(r, file)]
                        if ".txt" in file:
                           self.file[file.split(".")[0]] = [os.path.join(r, file)]   
                           
                           
D = DATA()
D.parseIMG("/home/sadko/images/_plate")  
img_file = D.file 

G = DATA()
G.parseIMG("/home/sadko/labels/plate") 
lable_file = G.file
 
for i in img_file:
    img = cv2.imread(img_file[i][0])
    #imgs(img)
    F = open(lable_file[i][0],"r")
    for o in F:
        s_str = o.split("\n")[0]
        coord = [float(a) for a in s_str.split(" ")[1:]]
        draw_boxes(img, coord)
    imgs(img)
    #print (F)#(img_file[i][0], lable_file[i][0])                                            
