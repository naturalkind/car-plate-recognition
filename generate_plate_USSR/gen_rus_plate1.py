#-*- coding: utf-8 -*-
# ГОСТ Р 50577-2018 http://docs.cntd.ru/document/1200160380
# auto

from PIL import Image, ImageFont, ImageDraw
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob, random
import json
from io import BytesIO

def imgs(x):
      cv2.imshow('Rotat', np.array(x))
      cv2.waitKey(0)
      cv2.destroyAllWindows()

class DATA(object):
   def __init__(self):
       self.file = {}
       self.label = {}

   def parseIMG(self, dir_name):
       path = dir_name+"/"
       print ("PARSING",path)
       for r, d, f in os.walk(path):
           for ix, file in enumerate(f[:20000]):#[:20000]
                           #print (file)
                      if ".png" in file: #".jpg" or 
                          img = cv2.imread(os.path.join(r, file))
                          self.file[file.split(".")[0]] = [os.path.join(r, file), img]
                      if ".jpg" in file: #".jpg" or 
                          img = cv2.imread(os.path.join(r, file))
                          self.file[file.split(".")[0]] = [os.path.join(r, file), img]    
                      if ".json" in file:
                          #jsf = open(os.path.join(r, file), 'r')
                          self.label[file.split(".")[0]] = [os.path.join(r, file)]
#spec = {"size": (112,520),
#        "len_text": 6,
#        "height":[]}



class GenPlate():
    def __init__(self):
        self.font = ImageFont.truetype("fonts/RoadNumbers2.0.ttf", size=500)
        self.plate = np.ones((112,520))
        self.PADDING_H = 36
        self.H_WORD = 58
        self.H_NUM = 76
        self.PADDING_LEFT = 41
        self.t = ""
        for ix in range(0, self.plate.shape[0]):
            for iy in range(self.plate.shape[1]): 
                if self.plate[ix,iy] != 0:
                   self.plate[ix,iy] = 255
        self.LETTER = "ABEKMHOPCTYX"
        self.DIGIT = "0123456789"
    def gen(self, o, p, x_p):
        (width, baseline), (offset_x, offset_y) = self.font.font.getsize(self.t)
        img = Image.new("RGBA", (width, baseline), (255,255,255))     
        draw = ImageDraw.Draw(img)  
        draw.text((abs(offset_x), abs(offset_y)), self.t, (0,0,0), font=self.font)
        shape = np.array(img).shape
        h = shape[0]
        w = shape[1]
        kef = o/h
        x_x = self.font.getmask(self.t).getbbox()[0]
        x_x = int(x_x * kef)
        img = cv2.resize(np.array(img),(int(w*kef),o))#, interpolation=cv2.INTER_AREA)
        self.plate[p:(p+o), (self.PADDING_LEFT-x_x):((self.PADDING_LEFT-x_x)+int(w*kef))] = img[:,:,0]
        self.PADDING_LEFT += (img.shape[1]-self.font.getmask(self.t).getbbox()[0]//2)
        self.PADDING_LEFT += x_p
        
    def gen_text_plate(self):
        
        self.t = random.choice(self.LETTER)
        self.gen(test.H_WORD, test.PADDING_H, 21)

        self.t = random.choice(self.DIGIT)#"9"
        self.gen(test.H_NUM, test.PADDING_H//2, 11)

        self.t = random.choice(self.DIGIT)#"7"
        self.gen(test.H_NUM, test.PADDING_H//2, 11)

        self.t = random.choice(self.DIGIT)#"6"
        self.gen(test.H_NUM, test.PADDING_H//2, 21)

        self.t = random.choice(self.LETTER)#"m"
        self.gen(test.H_WORD, test.PADDING_H, 11)

        self.t = random.choice(self.LETTER)#"m"
        self.gen(test.H_WORD, test.PADDING_H, 11)
#imgs(test.plate)
#plt.imshow(test.plate)
#plt.show()     

#print (test.plate[ix,iy])#(iy, ix, test.plate.shape)    
test = GenPlate()
test.gen_text_plate()
plt.imshow(test.plate)
plt.show()
#imgs(test.plate)


# отступ с низу 18
# 112-76=36 
# с снизу 36/2 = 18
# 112-(58+18) = 36

#58 	Толщина: 9,0
#76 	Толщина: 11,0

# Отступ между цифр (x ЩИФРФ xx) - 11.0 ~ 10.0
# Отступ от первого и последнего символа - 21.0 ~ 20.0

