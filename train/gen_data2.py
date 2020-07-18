#-*- coding: utf-8 -*-
#
from PIL import Image, ImageFont, ImageDraw
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

#text = "АВЕКМНОРСТУХ"
fl = "auto0.txt"
LETTER = "ABEKMHOPCTYX"
DIGIT = "0123456789"

#img = Image.new("RGBA", (128, 64), (120,20,20))
#img = Image.new("RGBA", (128, 64))
#draw = ImageDraw.Draw(img)
def gen_text():
   t = ""
   for ix in range(7):
      t += random.choice(list(LETTER))
   return t
      
#s = gen_text()      
#draw.text((0,0), s, (255,255,0), font=font)
#draw = ImageDraw.Draw(img)
image = cv2.imread("i/auto0.png")
#img.save("a_test.png")

#imgs(image) 
F = open(fl, "r")#.read()
#LRR = "C227HA"
#LRR = "C287HA"
LRR = "123745"
#LRR = "C219HA"
for iX, i in enumerate(F):
    cord = [int(u) for u in i.split("\n")[0].split(" ")[1:]]
    shape = image[cord[1]:cord[3], cord[0]:cord[2], :].shape
    font = ImageFont.truetype("fonts/RoadNumbers2.0.ttf", 500)
    #font = ImageFont.truetype("fonts/UKNumberPlate.ttf", 500)
    #t = random.choice(list(LETTER))
    t = list(LRR)[iX]
    #font.getsize(t)
    print (font.getsize(t)) 
#    new = np.ones(shape=shape, 
#                  dtype=np.float32)
    
    img = Image.new("RGBA", font.getsize(t), (255,255,255))
    draw = ImageDraw.Draw(img)             
    draw.text((0,0), t, (0,0,0), font=font)
    draw = ImageDraw.Draw(img)  
    #img.show()  
    resize = cv2.resize(np.array(img), (shape[1], shape[0]))     
    #imgs(resize) 
    print (resize.shape) 
    #img.resize((cord[2]-cord[0], cord[3]-cord[1]), Image.ANTIALIAS)
    #image[cord[1]:cord[3], cord[0]:cord[2], :] = new#2#55
    image[cord[1]:cord[3], cord[0]:cord[2], :] = resize[:,:,:3]
    print (cord, shape)
    
imgs(image)  
#http://avto-nomer.ru/xx/gallery.php?ctype=1 
#http://avto-nomer.ru/xx/gallery.php?&ctype=1&start=3
#http://avto-nomer.ru/ua/gallery
#image = Image.open('hsvwheel.png')
#draw = ImageDraw.Draw(image)
#txt = "Hello World"
#fontsize = 1  # starting font size

## portion of image width you want text width to be
#img_fraction = 0.50

#font = ImageFont.truetype("arial.ttf", fontsize)
#while font.getsize(txt)[0] < img_fraction*image.size[0]:
#    # iterate until the text size is just larger than the criteria
#    fontsize += 1
#    font = ImageFont.truetype("arial.ttf", fontsize)

## optionally de-increment to be sure it is less than criteria
#fontsize -= 1
#font = ImageFont.truetype("arial.ttf", fontsize)

#print 'final font size',fontsize
#draw.text((10, 25), txt, font=font) # put the text on the image
#image.save('hsvwheel_txt.png') # save it
