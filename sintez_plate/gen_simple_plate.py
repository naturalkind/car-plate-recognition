import itertools
import math
import os
import random
import sys
import json
import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import matplotlib.pyplot as plt
import uuid

OUTPUT_SHAPE = (64, 128)

    

def imgs(x):
      cv2.imshow('Rotat', np.array(x))
      cv2.waitKey(0)
      cv2.destroyAllWindows()


class DATA(object):
   def __init__(self):
       self.file = {}
       self.label = {}

   def parseIMG(self, dir_name):
       path = f"{dir_name}/"
       print ("PARSING",path)
       for r, d, f in os.walk(path):
           for ix, file in enumerate(f):
                      if ".png" in file:
                          self.file[file.split(".")[0]] = [os.path.join(r, file)]
                      if ".jpg" in file: 
                          self.file[file.split(".")[0]] = [os.path.join(r, file)]
                      if ".json" in file:
                          self.label[file.split(".")[0]] = [os.path.join(r, file)]
   def createARR(self, shape=(64, 128)):
       arr = []
       for i in self.file:
           im = cv2.imread(self.file[i][0])
           im = cv2.resize(im, shape).astype(np.float32) / 255.
           arr.append(im)
       return np.array(arr)

def gen(name_file, text="c227ha34"):
    OUTPUT_SHAPE = (1024, 512)
    font = ImageFont.truetype("fonts/RoadNumbers2.0.ttf", size=500)
    (width, baseline), (offset_x, offset_y) = font.font.getsize(text)
    img = Image.new("RGBA", (width+10, baseline+10), (255,255,255))
    draw = ImageDraw.Draw(img)
    draw.text((abs(offset_x), abs(offset_y)+5), text, (0,0,0), font=font)
    img = cv2.resize(np.array(img), OUTPUT_SHAPE)
    cv2.imwrite(name_file, img)
    
    #imgs(img)
    
    #plt.imshow(img)
    #plt.show()     
#gen()

D = DATA()
D.parseIMG("GEN_data_v2")

for i in D.file:
    text = D.file[i][0].split("/")[-1].split("_")[-1].split(".")[0]
    name_file = f'GEN_sample_char/{D.file[i][0].split("/")[-1]}'
    print (text, name_file)
    gen(name_file, text)
#    print (D.file[i][0], text, name_file)


#class GenPlate():
#    def __init__(self):
#        self.font = ImageFont.truetype("fonts/RoadNumbers2.0.ttf", size=500)
#        self.plate = 0
#        self.PADDING_H = 36
#        self.H_WORD = 58
#        self.H_NUM = 76
#        self.PADDING_LEFT = 41
#        self.t = ""
#        self.full_text = ""
#        self.LETTER = "ABEKMHOPCTYX"
#        self.DIGIT = "0123456789"
#    def gen(self, o, p, x_p):
#        (width, baseline), (offset_x, offset_y) = self.font.font.getsize(self.t)
#        img = Image.new("RGBA", (width, baseline), (255,255,255))     
#        draw = ImageDraw.Draw(img)  
#        draw.text((abs(offset_x), abs(offset_y)), self.t, (0,0,0), font=self.font)
#        shape = np.array(img).shape
#        h = shape[0]
#        w = shape[1]
#        kef = o/h
#        x_x = self.font.getmask(self.t).getbbox()[0]
#        x_x = int(x_x * kef)
#        dim = (int(w*kef),o)
#        #print (dim, self.t, x_x)
#        img = cv2.resize(np.array(img),dim)#, interpolation=cv2.INTER_AREA)
#        #imgs(img)
#        self.plate[p:(p+o), (self.PADDING_LEFT-x_x):((self.PADDING_LEFT-x_x)+int(w*kef)),:] = img[:,:,:3]
#        self.PADDING_LEFT += (img.shape[1]-self.font.getmask(self.t).getbbox()[0]//2)
#        self.PADDING_LEFT += x_p
#    def clear_plate(self):
#        plate = np.ones((112,520, 3))
#        for ix in range(plate.shape[0]):
#            for iy in range(plate.shape[1]): 
#                if plate[ix,iy,0] != 0:
#                     plate[ix,iy,0] = 255
#                     plate[ix,iy,1] = 255
#                     plate[ix,iy,2] = 255
#        return plate             
#        
#    def gen_text_plate(self):
#        self.plate = self.clear_plate()#np.ones((112,520))
#        self.full_text = ""
#        self.t = random.choice(self.LETTER)
#        self.full_text += self.t
#        self.gen(test.H_WORD, test.PADDING_H, 21)

#        self.t = random.choice(self.DIGIT)#"9"
#        self.full_text += self.t
#        self.gen(test.H_NUM, test.PADDING_H//2, 11)

#        self.t = random.choice(self.DIGIT)#"7"
#        self.full_text += self.t
#        self.gen(test.H_NUM, test.PADDING_H//2, 11)

#        self.t = random.choice(self.DIGIT)#"6"
#        self.full_text += self.t
#        self.gen(test.H_NUM, test.PADDING_H//2, 21)

#        self.t = random.choice(self.LETTER)#"m"
#        self.full_text += self.t
#        self.gen(test.H_WORD, test.PADDING_H, 11)

#        self.t = random.choice(self.LETTER)#"m"
#        self.full_text += self.t
#        self.gen(test.H_WORD, test.PADDING_H, 11)
#        
#        self.PADDING_LEFT = 41
#        
#    def text_plate(self, n_text):
#        self.plate = self.clear_plate()#np.ones((112,520))
#        self.full_text = ""
#        self.t = n_text[0]
#        self.full_text += self.t
#        self.gen(test.H_WORD, test.PADDING_H, 21)

#        self.t = n_text[1]#"9"
#        self.full_text += self.t
#        self.gen(test.H_NUM, test.PADDING_H//2, 11)

#        self.t = n_text[2]#"7"
#        self.full_text += self.t
#        self.gen(test.H_NUM, test.PADDING_H//2, 11)

#        self.t = n_text[3]#"6"
#        self.full_text += self.t
#        self.gen(test.H_NUM, test.PADDING_H//2, 21)

#        self.t = n_text[4]#"m"
#        self.full_text += self.t
#        self.gen(test.H_WORD, test.PADDING_H, 11)

#        self.t = n_text[5]#"m"
#        self.full_text += self.t
#        self.gen(test.H_WORD, test.PADDING_H, 11)
#        
#        self.PADDING_LEFT = 41


#class DATA(object):
#   def __init__(self):
#       self.file = {}
#       self.label = {}

#   def parseIMG(self, dir_name):
#       path = dir_name+"/"
#       print ("PARSING",path)
#       for r, d, f in os.walk(path):
#           for ix, file in enumerate(f):
#                      if ".png" in file:
#                          self.file[file.split(".")[0]] = [os.path.join(r, file)]
#                      if ".jpg" in file: 
#                          self.file[file.split(".")[0]] = [os.path.join(r, file)]
#                      if ".json" in file:
#                          self.label[file.split(".")[0]] = [os.path.join(r, file)]
#   def createARR(self, shape=(64, 128)):
#       arr = []
#       for i in self.file:
#           im = cv2.imread(self.file[i][0])
#           im = cv2.resize(im, shape).astype(np.float32) / 255.
#           arr.append(im)
#       return np.array(arr)


#def euler_to_mat(yaw, pitch, roll):
#    # Rotate clockwise about the Y-axis
#    c, s = math.cos(yaw), math.sin(yaw)
#    M = np.matrix([[  c, 0.,  s],
#                      [ 0., 1., 0.],
#                      [ -s, 0.,  c]])

#    # Rotate clockwise about the X-axis
#    c, s = math.cos(pitch), math.sin(pitch)
#    M = np.matrix([[ 1., 0., 0.],
#                      [ 0.,  c, -s],
#                      [ 0.,  s,  c]]) * M

#    # Rotate clockwise about the Z-axis
#    c, s = math.cos(roll), math.sin(roll)
#    M = np.matrix([[  c, -s, 0.],
#                      [  s,  c, 0.],
#                      [ 0., 0., 1.]]) * M

#    return M

#def make_affine_transform(from_shape, to_shape, 
#                          min_scale, max_scale,
#                          scale_variation=1.0,
#                          rotation_variation=1.0,
#                          translation_variation=1.0):
#    out_of_bounds = False

#    from_size = np.array([[from_shape[1], from_shape[0]]]).T
#    to_size = np.array([[to_shape[1], to_shape[0]]]).T

#    scale = random.uniform((min_scale + max_scale) * 0.5 -
#                           (max_scale - min_scale) * 0.5 * scale_variation,
#                           (min_scale + max_scale) * 0.5 +
#                           (max_scale - min_scale) * 0.5 * scale_variation)
#    if scale > max_scale or scale < min_scale:
#        out_of_bounds = True
#    roll = random.uniform(-0.3, 0.3) * rotation_variation
#    pitch = random.uniform(-0.2, 0.2) * rotation_variation
#    yaw = random.uniform(-1.2, 1.2) * rotation_variation

#    # Compute a bounding box on the skewed input image (`from_shape`).
#    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
#    h, w = from_shape
#    corners = np.matrix([[-w, +w, -w, +w],
#                            [-h, -h, +h, +h]]) * 0.5
#    skewed_size = np.array(np.max(M * corners, axis=1) -
#                              np.min(M * corners, axis=1))

#    # Set the scale as large as possible such that the skewed and scaled shape
#    # is less than or equal to the desired ratio in either dimension.
#    scale *= np.min(to_size / skewed_size)

#    # Set the translation such that the skewed and scaled image falls within
#    # the output shape's bounds.
#    trans = (np.random.random((2,1)) - 0.5) * translation_variation
#    trans = ((2.0 * trans) ** 5.0) / 2.0
#    if np.any(trans < -0.5) or np.any(trans > 0.5):
#        out_of_bounds = True
#    trans = (to_size - skewed_size * scale) * trans

#    center_to = to_size / 2.
#    center_from = from_size / 2.

#    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
#    M *= scale
#    M = np.hstack([M, trans + center_to - M * center_from])

#    return M, out_of_bounds

#def generate_bg():
#    found = False
#    while not found:
#        fname = random.choice(list(BG.file.keys()))
#        bg = cv2.imread(BG.file[fname][0], cv2.IMREAD_GRAYSCALE) / 255.
#        if (bg.shape[1] >= OUTPUT_SHAPE[1] and
#            bg.shape[0] >= OUTPUT_SHAPE[0]):
#            found = True
#    x = random.randint(0, bg.shape[1] - OUTPUT_SHAPE[1])
#    y = random.randint(0, bg.shape[0] - OUTPUT_SHAPE[0])
#    bg = bg[y:y + OUTPUT_SHAPE[0], x:x + OUTPUT_SHAPE[1]]
#    return bg


#def generate_im(plate, code):
#    bg = generate_bg()
#    plate_mask = np.ones(plate.shape)
#    M, out_of_bounds = make_affine_transform(
#                            from_shape=plate.shape,
#                            to_shape=bg.shape,
#                            min_scale=0.6,
#                            max_scale=0.875,
#                            rotation_variation=1.0,
#                            scale_variation=1.5,
#                            translation_variation=1.2)
#    plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
#    plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))
#    out = (plate * plate_mask)/255. + bg * (1.0 - plate_mask)
#    out = cv2.resize(out, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))
#    out += np.random.normal(scale=0.05, size=out.shape)
#    out = np.clip(out, 0., 1.)
#    return out, code, not out_of_bounds


#def generate_ims_folder():
#    while True:
#        j = random.choice(P_IMG)
#        text = j.split("_")[-1].split(".")[0]
#        img = cv2.imread(IMG.file[j][0])
#        yield generate_im(img[:,:,0], text)

#def generate_ims_aff():
#    while True:
#        test.gen_text_plate()
#        test.plate = gan.show_predict(test.plate)
#        test.plate = cv2.resize(test.plate, (128,64))
#        yield generate_im(test.plate[:,:,0]*255., test.full_text)

#def generate_ims():
#    while True:
#        test.gen_text_plate()
#        yield test.plate, test.full_text


#BG = DATA()
#BG.parseIMG("bgs")
#test = GenPlate()
#IMG = DATA()
#IMG.parseIMG("real")
#P_IMG = list(IMG.file.keys())


#if __name__ == "__main__":
#    
#    if sys.argv[1] == "non_gen":
#        im_gen = itertools.islice(generate_ims_folder(), int(sys.argv[2]))
#        for img_idx, (im, c, p) in enumerate(im_gen):
##            imgs(im)
#            fname = "test/{:08d}_{}_{}.png".format(img_idx, c,
#                                                           "1" if p else "0")
#            cv2.imwrite(fname, im * 255.) 
#    if sys.argv[1] == "gen_aff":
#        im_gen = itertools.islice(generate_ims_aff(), int(sys.argv[2]))
#        for img_idx, (im, c, p) in enumerate(im_gen):
##             imgs(im)
#            fname = "test/{:08d}_{}_{}.png".format(img_idx, c,
#                                                           "1" if p else "0")
#            cv2.imwrite(fname, im * 255.) 
#              
#    if sys.argv[1] == "gen":
#        im_gen = itertools.islice(generate_ims(), int(sys.argv[2]))
#        for img_idx, (im, c) in enumerate(im_gen):
##            imgs(im)  
##            plt.imshow(im)
##             plt.show()  
#              cv2.imwrite(f"{c}.png", im * 255.) 

