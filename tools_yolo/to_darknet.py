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
    #print img_h, img_w, _
    #print box #g = [a for a in box]
    #box = [a for a in box]
    # x1, y1 = ((x + witdth)/2)*img_width, ((y + height)/2)*img_height
    # x2, y2 = ((x - witdth)/2)*img_width, ((y - height)/2)*img_height
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
    imgs(img)
    #return img
#------------------------------------------------------>
def convert(size, box):
   dw = 1./size[0]
   dh = 1./size[1]
   x = (box[0] + box[1])/2.0
   y = (box[2] + box[3])/2.0
   w = box[1] - box[0]
   h = box[3] - box[2]
   x = x*dw
   w = w*dw
   y = y*dh
   h = h*dh
   return (x,y,w,h)


def crt(files, idx, dq):
    print (files, idx, dq)
    F = open('tr'+str(idx)+'.txt', 'w')
    for ig, g in enumerate(files):
            
            with open("Labels/"+g+".txt", 'r') as f:
                f = f.readlines()
                if len(f) != 0:
                    
                    img = cv2.imread(str(dq[g][0]))
                    to_vl = '/home/sadko/images/plate/'+dq[g][0].split("/")[-1]
#                   #flopen = open(str(dq[g][0]), "rb").read()
#                   #flsave = open(to_vl, "wb").write(flopen)
                    cv2.imwrite(to_vl, img)
                    F.write(to_vl +'\n')  
                    xml = ''
                    for o in f:
                       print (dq[g][0], o)
                       gh = o.split("\n")[0]
                       C = gh.split(" ")[0]
                       gh = gh.split(" ")[1:]
                       w = int(img.shape[1])#int(416)
                       h = int(img.shape[0])#int(416)
                       b = (float(gh[0]), float(gh[2]),float(gh[1]), float(gh[3]))           
                       bb = convert((w,h), b)       

                       xml += f'{CLASS.index(C)} ' + ' '.join([str(a) for a in bb]) + '\n'
#                       draw_boxes(img, bb)
                    fl = open('TRAIN_LABEL/' + g + '.txt', 'w')
                    fl.write(xml)
                    fl.close()
    F.close()




#files = glob.glob("Labels/*")
#    
#random.shuffle(files)
#ln = len(files) / 100 * 11
#print len(files), ln

#newkey, newkey1 = crt(files[:ln], 0), crt(files[ln:], 1)  
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

if __name__ == '__main__':
    D = DATA()
    D.parseIMG("srgan_orig_predict")
    #print (D.file.values())
    
    ln = int(len(list(D.file.keys())) / 100 * 5)
    print (ln)
    CLASS = open("class.txt", "r")#.read()
    CLASS = [i.split("\n")[0] for i in CLASS]
    print (CLASS)
    
        
    newkey, newkey1 = crt(list(D.file.keys())[:ln], 0, D.file), crt(list(D.file.keys())[ln:], 1, D.file) 
    #print ln

