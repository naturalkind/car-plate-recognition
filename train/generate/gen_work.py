#!/usr/bin/env python
#
# Copyright (c) 2016 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.



"""
Generate training and test images.

"""


__all__ = (
    'generate_ims',
)


import itertools
import math
import os
import random
import sys
import json
import cv2
import numpy

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import common

FONT_DIR = "./fonts"
FONT_HEIGHT = 32  # Pixel size to which the chars are resized

OUTPUT_SHAPE = (64, 128)

CHARS = common.CHARS + " "

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
           for ix, file in enumerate(f):
                           #print (file)
                      if ".png" in file: #".jpg" or 
                          self.file[file.split(".")[0]] = [os.path.join(r, file)]
                      if ".json" in file:
                          self.label[file.split(".")[0]] = [os.path.join(r, file)]


def make_char_ims(font_path, output_height):
    font_size = output_height * 4

    font = ImageFont.truetype(font_path, font_size)

    height = max(font.getsize(c)[1] for c in CHARS)

    for c in CHARS:
        width = font.getsize(c)[0]
        im = Image.new("RGBA", (width, height), (0, 0, 0))

        draw = ImageDraw.Draw(im)
        draw.text((0, 0), c, (255, 255, 255), font=font)
        scale = float(output_height) / height
        im = im.resize((int(width * scale), output_height), Image.ANTIALIAS)
        yield c, numpy.array(im)[:, :, 0].astype(numpy.float32) / 255.


def euler_to_mat(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = numpy.matrix([[  c, 0.,  s],
                      [ 0., 1., 0.],
                      [ -s, 0.,  c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = numpy.matrix([[ 1., 0., 0.],
                      [ 0.,  c, -s],
                      [ 0.,  s,  c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = numpy.matrix([[  c, -s, 0.],
                      [  s,  c, 0.],
                      [ 0., 0., 1.]]) * M

    return M


def pick_colors():
    first = True
    while first or plate_color - text_color < 0.3:
        text_color = random.random()
        plate_color = random.random()
        if text_color > plate_color:
            text_color, plate_color = plate_color, text_color
        first = False
    return text_color, plate_color


def make_affine_transform(from_shape, to_shape, 
                          min_scale, max_scale,
                          scale_variation=1.0,
                          rotation_variation=1.0,
                          translation_variation=1.0):
    out_of_bounds = False

    from_size = numpy.array([[from_shape[1], from_shape[0]]]).T
    to_size = numpy.array([[to_shape[1], to_shape[0]]]).T

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
    corners = numpy.matrix([[-w, +w, -w, +w],
                            [-h, -h, +h, +h]]) * 0.5
    skewed_size = numpy.array(numpy.max(M * corners, axis=1) -
                              numpy.min(M * corners, axis=1))

    # Set the scale as large as possible such that the skewed and scaled shape
    # is less than or equal to the desired ratio in either dimension.
    scale *= numpy.min(to_size / skewed_size)

    # Set the translation such that the skewed and scaled image falls within
    # the output shape's bounds.
    trans = (numpy.random.random((2,1)) - 0.5) * translation_variation
    trans = ((2.0 * trans) ** 5.0) / 2.0
    if numpy.any(trans < -0.5) or numpy.any(trans > 0.5):
        out_of_bounds = True
    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.
    center_from = from_size / 2.

    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    M *= scale
    M = numpy.hstack([M, trans + center_to - M * center_from])

    return M, out_of_bounds


def generate_code():
    return "{}{}{}{} {}{}{}".format(
        random.choice(common.LETTERS),
        random.choice(common.LETTERS),
        random.choice(common.DIGITS),
        random.choice(common.DIGITS),
        random.choice(common.LETTERS),
        random.choice(common.LETTERS),
        random.choice(common.LETTERS))


def rounded_rect(shape, radius):
    out = numpy.ones(shape)
    out[:radius, :radius] = 0.0
    out[-radius:, :radius] = 0.0
    out[:radius, -radius:] = 0.0
    out[-radius:, -radius:] = 0.0

    cv2.circle(out, (radius, radius), radius, 1.0, -1)
    cv2.circle(out, (radius, shape[0] - radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, shape[0] - radius), radius, 1.0, -1)

    return out


def generate_plate(font_height, char_ims):
    h_padding = random.uniform(0.2, 0.4) * font_height
    v_padding = random.uniform(0.1, 0.3) * font_height
    spacing = font_height * random.uniform(-0.05, 0.05)
    radius = 1 + int(font_height * 0.1 * random.random())

    code = generate_code()
    text_width = sum(char_ims[c].shape[1] for c in code)
    text_width += (len(code) - 1) * spacing

    out_shape = (int(font_height + v_padding * 2),
                 int(text_width + h_padding * 2))

    text_color, plate_color = pick_colors()
    
    text_mask = numpy.zeros(out_shape)
    
    x = h_padding
    y = v_padding 
    for c in code:
        char_im = char_ims[c]
        ix, iy = int(x), int(y)
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
        x += char_im.shape[1] + spacing

    plate = (numpy.ones(out_shape) * plate_color * (1. - text_mask) +
             numpy.ones(out_shape) * text_color * text_mask)

    return plate, rounded_rect(out_shape, radius), code.replace(" ", "")


def generate_bg(num_bg_images):
    found = False
    while not found:
        fname = "bgs/{:08d}.jpg".format(random.randint(0, num_bg_images - 1))
        bg = cv2.imread(fname, cv2.IMREAD_GRAYSCALE) / 255.
        if (bg.shape[1] >= OUTPUT_SHAPE[1] and
            bg.shape[0] >= OUTPUT_SHAPE[0]):
            found = True

    x = random.randint(0, bg.shape[1] - OUTPUT_SHAPE[1])
    y = random.randint(0, bg.shape[0] - OUTPUT_SHAPE[0])
    bg = bg[y:y + OUTPUT_SHAPE[0], x:x + OUTPUT_SHAPE[1]]

    return bg

def imgs(x):
      cv2.imshow('Rotat', numpy.array(x))
      cv2.waitKey(0)
      cv2.destroyAllWindows()

def generate_im(char_ims, num_bg_images, code):
    bg = num_bg_images*255#generate_bg(num_bg_images)
    bg = bg.astype(numpy.uint8)
    #plate, plate_mask, code
    plate = char_ims.astype(numpy.uint8)
    plate_mask_t = numpy.ones(char_ims.shape)
    
    #
    plate = char_ims#generate_plate(FONT_HEIGHT, char_ims)
    #print (type(plate), type(plate_mask), plate.shape, plate_mask.shape, code)
    #print (type(plate), type(num_bg_images), plate.shape, num_bg_images.shape, code)

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
    #imgs(plate_mask)
    #imgs(plate)
    #out = plate * plate_mask + bg * plate_mask
    #out = plate * plate_mask + bg * (1.0 - plate_mask)
    out = plate + bg# * 255
    for iw in range(plate_mask.shape[0]):
        for ih in range(plate_mask.shape[1]):
            if plate_mask[iw,ih] == 1.0:
               bg[iw,ih] = plate[iw,ih]
               #print (plate_mask[iw,ih])
      
    #imgs(bg)
    out = cv2.resize(bg, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))# data.astype(np.uint8)#/.255
    #print (out.shape)
    #out += numpy.random.normal(scale=0.05, size=out.shape)
    #out = numpy.clip(out, 0., 1.)
    #return plate_mask, code, not out_of_bounds
    return out, code, not out_of_bounds


def load_fonts(folder_path):
    font_char_ims = {}
    fonts = [f for f in os.listdir(folder_path) if f.endswith('.ttf')]
    for font in fonts:
        font_char_ims[font] = dict(make_char_ims(os.path.join(folder_path,
                                                              font),
                                                 FONT_HEIGHT))
    return fonts, font_char_ims



H = DATA()
H.parseIMG("data_ru")

H1 = DATA()
H1.parseIMG("data_plate_nomeroff")

H2 = DATA()
H2.parseIMG("data_plate_nomeroff1")
#print (H.label.keys(), len(H.file.keys()))
#print (len(H.file.keys()))
num_bg_images = len(os.listdir("bgs"))


#for ikk in range(1000):
#        j = random.choice(list(H.label.keys()))
#        fl_open = open(H.label[j][0], "r")
#        json_open = json.load(fl_open)
#        print (json_open["description"], json_open["size"])
#        bg_img = gen2.generate_bg(num_bg_images)
#        img = cv2.imread(H.file[j][0])
#        random_scale = random.choice([2, 2.3])
#        #int((json_open["size"]["width"]*128)/1000
#        #
#        #print (bg_img.shape, img_res.shape) #img_res[:,:,0]
#        #imgs(bg_img)

#        tn_img = gen2.generate_im(img[:,:,0], np.array(bg_img))
#        imgs(tn_img[0])
#        print (tn_img[1:], json_open["description"])

import gen as G0 #original code
import gen1 as G1 #original full chars
import gen2 as G2 #original full chars
import gen3 as G3 #original full chars
import gen4 as G4 #original evolu code
def generate_ims():
    """
    Generate number plate images.

    :return:
        Iterable of number plate images.

    """
    variation = 1.0
    fonts, font_char_ims = load_fonts(FONT_DIR)
    #fonts1, font_char_ims1 = load_fonts("fonts_")
    #print (fonts1)
    num_bg_images = len(os.listdir("bgs"))
    while True:

        #------------>

        
        bg_img = generate_bg(num_bg_images)
        
        
        #print (json_open["description"], H1.label.keys())
        ls_t = [0,1,2,3,4,5,6]
        ls_t = random.choice(ls_t)
        if ls_t == 0:
           yield G0.generate_im(font_char_ims[random.choice(fonts)], num_bg_images)
        if ls_t == 1:
           yield G4.generate_im(font_char_ims[random.choice(fonts)], num_bg_images)
        if ls_t == 2:
           yield G1.generate_im(font_char_ims[random.choice(fonts)], num_bg_images)
        if ls_t == 3:
           yield G3.generate_im(font_char_ims[random.choice(fonts)], num_bg_images)


        if ls_t == 4:
           j = random.choice(list(H.label.keys()))
           json_open = json.load(open(H.label[j][0], "r"))
           img = cv2.imread(H.file[j][0])
           yield generate_im(img[:,:,0], numpy.array(bg_img), json_open["description"])
        if ls_t == 5:
           j1 = random.choice(list(H1.label.keys()))
           json_open1 = json.load(open(H1.label[j1][0], "r"))
           img1 = cv2.imread(H1.file[j1][0])
           yield G2.generate_im(img1[:,:,0], numpy.array(bg_img), json_open1["description"])
        if ls_t == 6:
           j2 = random.choice(list(H2.label.keys()))
           json_open2 = json.load(open(H2.label[j2][0], "r"))
           img2 = cv2.imread(H2.file[j2][0])
           yield G2.generate_im(img2[:,:,0], numpy.array(bg_img), json_open2["description"])



        #yield generate_im(img[:,:,0], numpy.array(bg_img), json_open["description"])


def generate_ims1():
    """
    Generate number plate images.

    :return:
        Iterable of number plate images.

    """
    variation = 1.0
    fonts, font_char_ims = load_fonts(FONT_DIR)
    num_bg_images = len(os.listdir("bgs"))
    while True:
        yield generate_im(num_bg_images)


if __name__ == "__main__":
    os.mkdir("test")
    im_gen = itertools.islice(generate_ims(), int(sys.argv[1]))
    for img_idx, (im, c, p) in enumerate(im_gen):
        fname = "test/{:08d}_{}_{}.png".format(img_idx, c, "1" if p else "0")
        #print (fname)
        #imgs(im)
        
        cv2.imwrite(fname, im)

