#-*- coding: utf-8 -*-
#
from PIL import Image, ImageFont, ImageDraw
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob, random
import json
import math
import itertools
import sys
#from cairosvg import svg2png
from io import BytesIO

def imgs(x):
      cv2.imshow('Rotat', np.array(x))
      cv2.waitKey(0)
      cv2.destroyAllWindows()

class DATA_F(object):
   def __init__(self):
       self.file = {}

   def parseIMG(self, dir_name):
       path = dir_name+"/"
       print ("PARSING",path)
       for r, d, f in os.walk(path):
           for ix, file in enumerate(f):
                           #print (file)
                      if ".otf" in file or ".ttf" in file: 
                          self.file[file.split(".")[0]] = [os.path.join(r, file)]
                          
class DATA1(object):
   def __init__(self):
       self.file = {}
       self.label = {}

   def parseIMG(self, dir_name):
       path = dir_name+"/"
       print ("PARSING",path)
       for r, d, f in os.walk(path):
           for ix, file in enumerate(f):
                           #print (file)
                      if ".jpg" in file: #".jpg" or 
                          self.file[file.split(".")[0]] = [os.path.join(r, file)]
                      if ".json" in file:
                          self.label[file.split(".")[0]] = [os.path.join(r, file)]
                          
                                                    
def euler_to_mat(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = np.matrix([[  c, 0.,  s],
                      [ 0., 1., 0.],
                      [ -s, 0.,  c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = np.matrix([[ 1., 0., 0.],
                      [ 0.,  c, -s],
                      [ 0.,  s,  c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = np.matrix([[  c, -s, 0.],
                      [  s,  c, 0.],
                      [ 0., 0., 1.]]) * M

    return M

def make_affine_transform(from_shape, to_shape, 
                          min_scale, max_scale,
                          scale_variation=1.0,
                          rotation_variation=1.0,
                          translation_variation=1.0):
    out_of_bounds = False

    from_size = np.array([[from_shape[1], from_shape[0]]]).T
    to_size = np.array([[to_shape[1], to_shape[0]]]).T

    scale = random.uniform((min_scale + max_scale) * 0.5 -
                           (max_scale - min_scale) * 0.5 * scale_variation,
                           (min_scale + max_scale) * 0.5 +
                           (max_scale - min_scale) * 0.5 * scale_variation)
    if scale > max_scale or scale < min_scale:
        out_of_bounds = True
    roll = random.uniform(-0.3, 0.3) * rotation_variation
    pitch = random.uniform(-0.2, 0.2) * rotation_variation
    yaw = random.uniform(-1.2, 1.2) * rotation_variation

    # Compute a bounding box on the skewed input image (`from_shape`).
    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    h, w = from_shape
    corners = np.matrix([[-w, +w, -w, +w],
                            [-h, -h, +h, +h]]) * 0.5
    skewed_size = np.array(np.max(M * corners, axis=1) -
                              np.min(M * corners, axis=1))

    # Set the scale as large as possible such that the skewed and scaled shape
    # is less than or equal to the desired ratio in either dimension.
    scale *= np.min(to_size / skewed_size)

    # Set the translation such that the skewed and scaled image falls within
    # the output shape's bounds.
    trans = (np.random.random((2,1)) - 0.5) * translation_variation
    trans = ((2.0 * trans) ** 5.0) / 2.0
    if np.any(trans < -0.5) or np.any(trans > 0.5):
        out_of_bounds = True
    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.
    center_from = from_size / 2.

    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    M *= scale
    M = np.hstack([M, trans + center_to - M * center_from])

    return M, out_of_bounds 
 
        
def generate_im(char_ims, num_bg_images, code):
    bg = num_bg_images*255 #generate_bg(num_bg_images)
    bg = bg.astype(np.uint8)
    
    plate = char_ims.astype(np.uint8)
    plate_mask_t = np.ones(char_ims.shape)
    #
    plate = char_ims

    M, out_of_bounds = make_affine_transform(
                            from_shape=plate.shape,
                            to_shape=bg.shape,
                            min_scale=0.6,
                            max_scale=0.875,
                            rotation_variation=1.0,
                            scale_variation=1.5,
                            translation_variation=1.2)
    plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
    plate_mask = cv2.warpAffine(plate_mask_t, M, (bg.shape[1], bg.shape[0]))
    for iw in range(plate_mask.shape[0]):
        for ih in range(plate_mask.shape[1]):
            if plate_mask[iw,ih] == 1.0:
               bg[iw,ih] = plate[iw,ih]
               #print (plate_mask[iw,ih])
      
    #imgs(bg)
    out = cv2.resize(bg, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0])).astype(np.float32) / 255.# data.astype(np.uint8)#/.255
    return out, code, not out_of_bounds        
 
def generate_bg():
    found = False
    while not found:
        fname = random.choice(list(BG.file.keys()))
        bg = cv2.imread(BG.file[fname][0], cv2.IMREAD_GRAYSCALE) / 255.
        if (bg.shape[1] >= OUTPUT_SHAPE[1] and
            bg.shape[0] >= OUTPUT_SHAPE[0]):
            found = True

    x = random.randint(0, bg.shape[1] - OUTPUT_SHAPE[1])
    y = random.randint(0, bg.shape[0] - OUTPUT_SHAPE[0])
    bg = bg[y:y + OUTPUT_SHAPE[0], x:x + OUTPUT_SHAPE[1]]

    return bg

def generate_ims():
  while True:
          bg_img = generate_bg()
          platesx = plate_gen() 
          yield generate_im(platesx[0][:,:,0], np.array(bg_img), platesx[1])                       

fl = "auto1.txt"
LETTER = "ABEKMHOPCTYX"
DIGIT = "0123456789"
OUTPUT_SHAPE = (64, 128)
BG = DATA1()
BG.parseIMG("bgs")
#image = cv2.imread("i/Republic_2015.png")
#imgs(image)
#image = cv2.imread("i/plate0.png")
#image = cv2.resize(image, (1280,278))
F = open(fl, "r").readlines()
#(278, 1280, 3)
#fonts_dpr
FONTS = DATA_F()
FONTS.parseIMG("fonts/fonts_dpr")
#print (FONTS.file)
#, encoding="unic")
#первая буква отступ 85px >>>>>>>>>>>>>>>>>>> 110px
#первая буква и вторая цифра - 51px >>>>>>>>>> 63px
#25px
#25
#25
#50
#27px - отступ снизу для всех
#22px/43px - отступ сверху цифры
#66px/87px - отступ сверху буквы
#64px - отступ с лева
#Высота цифр 187px
#Высота букв 143px
#ширина букв 104
#ширина цифр 109
#210 276
def genrator_text():
   return "{}{}{}{}{}{}".format(random.choice(LETTER),random.choice(DIGIT), random.choice(DIGIT),random.choice(DIGIT),random.choice(LETTER),random.choice(LETTER))
#font = ImageFont.truetype("fonts/RoadNumbers2.0.ttf", 500)   
def plate_gen():
#for i in range(10):
        gjk = random.choice(list(FONTS.file.keys()))
        SZZ = random.choice([500,200, 64, 32, 100,400])
        font = ImageFont.truetype(FONTS.file[gjk][0], SZZ)
        NUMBER_TEXT = genrator_text()
        image = cv2.imread("i/Republic_.png")
        #cof = random.choice([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
        #print (cof, (cof/2))
        for iX, i in enumerate(F):
                cof = random.choice([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
                cord = [int(u) for u in i.split("\n")[0].split(" ")[1:]]
                #font = ImageFont.truetype("fonts/RoadNumbers2.0.ttf", 500)#, encoding="unic")
                #ascent, descent = font.getmetrics()
                t = NUMBER_TEXT[iX]
                (width, baseline), (offset_x, offset_y) = font.font.getsize(t)

                #print (cof, (cof/2), FONTS.file[gjk][0],(width, baseline), (offset_x, offset_y), font.getmask(t).getbbox(), font.getsize(t), font.font.getsize(t))
                #img = Image.new("RGBA", font.getmask("7").getbbox()[2:], (255,255,255))
                #img = Image.new("RGBA", (196, 319), (255,255,255))  
                img = Image.new("RGBA", (width, baseline), (255,255,255))   
                draw = ImageDraw.Draw(img)
                #draw.text((abs(offset_x), abs(offset_y)), t, (0,0,0), font=font)
                ZY = random.choice([-10, 10, abs(offset_y), 20, -1, 5,-5, offset_y])
                draw.text((0,(offset_y*-1)), t, (0,0,0), font=font)
                #imgs(img) 
                shape = image[cord[1]:cord[3], cord[0]:cord[2], :].shape
                resize = cv2.resize(np.array(img), (shape[1]+cof, shape[0]+cof))
                #print (image[cord[1]:cord[3], cord[0]:cord[2], :] )
                image[cord[1]-int(cof/2):cord[3]+int(cof/2), cord[0]-int(cof/2):cord[2]+int(cof/2), :] = resize[:,:,:3]
        #draw = ImageDraw.Draw(img)
        #imgs(image) 
        return image, NUMBER_TEXT
        
#platesx = plate_gen() 

#imgs(platesx[0])
if __name__ == "__main__":
    #os.mkdir("test")
    im_gen = itertools.islice(generate_ims(), int(sys.argv[1]))
    for img_idx, (im, c, p) in enumerate(im_gen):
        fname = "test/{:08d}_{}_{}.png".format(img_idx, c,
                                               "1" if p else "0")
        #cv2.imwrite(fname, im * 255.)                                       
        imgs(im)                                      
