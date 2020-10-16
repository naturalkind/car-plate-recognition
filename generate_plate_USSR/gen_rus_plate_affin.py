#-*- coding: utf-8 -*-
# ГОСТ Р 50577-2018 http://docs.cntd.ru/document/1200160380
# auto

from PIL import Image, ImageFont, ImageDraw
import os
import itertools
import sys
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob, random
import json
import math

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
BG = DATA()
BG.parseIMG("bgs")

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
    
    
class GenPlate():
    def __init__(self):
        self.font = ImageFont.truetype("fonts/RoadNumbers2.0.ttf", size=500)
        self.plate = 0
        self.PADDING_H = 36
        self.H_WORD = 58
        self.H_NUM = 76
        self.PADDING_LEFT = 41
        self.t = ""
        self.full_text = ""
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
        dim = (int(w*kef),o)
        #print (dim, self.t, x_x)
        img = cv2.resize(np.array(img),dim)#, interpolation=cv2.INTER_AREA)
        #imgs(img)
        self.plate[p:(p+o), (self.PADDING_LEFT-x_x):((self.PADDING_LEFT-x_x)+int(w*kef))] = img[:,:,0]
        self.PADDING_LEFT += (img.shape[1]-self.font.getmask(self.t).getbbox()[0]//2)
        self.PADDING_LEFT += x_p
    def clear_plate(self):
        plate = np.ones((112,520))
        for ix in range(plate.shape[0]):
            for iy in range(plate.shape[1]): 
                if plate[ix,iy] != 0:
                     plate[ix,iy] = 255
        return plate             
        
    def gen_text_plate(self):
        self.plate = self.clear_plate()#np.ones((112,520))
        self.full_text = ""
        self.t = random.choice(self.LETTER)
        self.full_text += self.t
        self.gen(test.H_WORD, test.PADDING_H, 21)

        self.t = random.choice(self.DIGIT)#"9"
        self.full_text += self.t
        self.gen(test.H_NUM, test.PADDING_H//2, 11)

        self.t = random.choice(self.DIGIT)#"7"
        self.full_text += self.t
        self.gen(test.H_NUM, test.PADDING_H//2, 11)

        self.t = random.choice(self.DIGIT)#"6"
        self.full_text += self.t
        self.gen(test.H_NUM, test.PADDING_H//2, 21)

        self.t = random.choice(self.LETTER)#"m"
        self.full_text += self.t
        self.gen(test.H_WORD, test.PADDING_H, 11)

        self.t = random.choice(self.LETTER)#"m"
        self.full_text += self.t
        self.gen(test.H_WORD, test.PADDING_H, 11)
        
        self.PADDING_LEFT = 41
OUTPUT_SHAPE = (64, 128)
test = GenPlate()

def generate_ims():
  while True:
          bg_img = generate_bg()
          test.gen_text_plate()#plate_gen() 
          yield generate_im(test.plate, np.array(bg_img), test.full_text)        


#imgs(test.plate)
#plt.imshow(test.plate)
#plt.show()     

#print (test.plate[ix,iy])#(iy, ix, test.plate.shape)    


#test.gen_text_plate()
    

if __name__ == "__main__":
    os.mkdir("test")
#    OUTPUT_SHAPE = (64, 128)
#    test = GenPlate()
#    #test.gen_text_plate()
#    
#    BG = DATA()
#    BG.parseIMG("bgs")
    im_gen = itertools.islice(generate_ims(), int(sys.argv[1]))
    for img_idx, (im, c, p) in enumerate(im_gen):
        #print (img_idx, c, p)
        fname = "test/{:08d}_{}_{}.png".format(img_idx, c,
                                               "1" if p else "0")
        #imgs(im)                                       
        cv2.imwrite(fname, im * 255.)    
#        for i in range(im.shape[0]):
#            for ii in range(im.shape[1]):    
#                if im[i,ii] != 0
#                   im[i,ii] = 255                           
#        imgs(im)  
#        plt.imshow(im)
#        plt.show()
#    plt.imshow(test.plate)
#    plt.show()
#    imgs(test.plate)


# отступ с низу 18
# 112-76=36 
# с снизу 36/2 = 18
# 112-(58+18) = 36

#58 	Толщина: 9,0
#76 	Толщина: 11,0

# Отступ между цифр (x ЩИФРФ xx) - 11.0 ~ 10.0
# Отступ от первого и последнего символа - 21.0 ~ 20.0

