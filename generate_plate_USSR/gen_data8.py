#-*- coding: utf-8 -*-
#
from PIL import Image, ImageFont, ImageDraw
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob, random
import json
#import svgwrite
from cairosvg import svg2png
from io import BytesIO

def imgs(x):
      cv2.imshow('Rotat', np.array(x))
      cv2.waitKey(0)
      cv2.destroyAllWindows()

fl = "auto0.txt"
LETTER = "ABEKMHOPCTYX"
DIGIT = "0123456789"

def gen_text():
   t = ""
   for ix in range(7):
      t += random.choice(list(LETTER))
   return t
   
#sS = cv2.imread("i/auto0.png")
#plt.imshow(sS)  
#plt.show() 
SVG  = open("i/Republic.svg","rb")
png = svg2png(bytestring=SVG.read())  
png = np.fromstring(png, np.uint8)
image = cv2.imdecode(png, cv2.IMREAD_COLOR)     
image = cv2.resize(image, (1280, 278))
#print (image)
F = open(fl, "r").readlines()
#(278, 1280, 3)


def genrator_text():
   return [random.choice(LETTER),random.choice(DIGIT), random.choice(DIGIT),random.choice(DIGIT),random.choice(LETTER),random.choice(LETTER)]

for i in range(10):
        NUMBER_TEXT = genrator_text()
        for iX, i in enumerate(F):
                cord = [int(u) for u in i.split("\n")[0].split(" ")[1:]]
                font = ImageFont.truetype("fonts/RoadNumbers2.0.ttf", 500)#, encoding="unic")
                #ascent, descent = font.getmetrics()
                t = NUMBER_TEXT[iX]
                (width, baseline), (offset_x, offset_y) = font.font.getsize(t)

                print ((width, baseline), (offset_x, offset_y), font.getmask(t).getbbox(), font.getsize(t))
                #img = Image.new("RGBA", font.getmask("7").getbbox()[2:], (255,255,255))
                #img = Image.new("RGBA", (196, 319), (255,255,255))  
                img = Image.new("RGBA", (width, baseline), (255,255,255))   
                draw = ImageDraw.Draw(img)
                draw.text((abs(offset_x), abs(offset_y)), t, (0,0,0), font=font)
                #imgs(img) 
                shape = image[cord[1]:cord[3], cord[0]:cord[2], :].shape
                resize = cv2.resize(np.array(img), (shape[1], shape[0]))
                image[cord[1]:cord[3], cord[0]:cord[2], :] = resize[:,:,:3]
        #draw = ImageDraw.Draw(img)
        imgs(image)  
