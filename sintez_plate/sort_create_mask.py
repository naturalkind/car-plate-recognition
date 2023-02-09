from __future__ import print_function, division
import numpy as np
import sys, cv2, os
import json, time, uuid
import random
import uuid


#https://gautamnagrawal.medium.com/rotating-image-by-any-angle-shear-transformation-using-only-numpy-d28d16eb5076
#https://www.pythoninformer.com/python-libraries/numpy/image-transforms/
#https://www.kaggle.com/irinaabdullaeva/text-segmentation
#https://github.com/SHI-Labs/Rethinking-Text-Segmentation
#https://habr.com/ru/company/intel/blog/266347/
#CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHARS = "ABEKMHIOPCTYX0123456789"


class DATA(object):
   def __init__(self):
       self.file = {}
       self.label = {}

   def parseIMG(self, dir_name):
       path = dir_name+"/"
       print ("PARSING",path)
       for r, d, f in os.walk(path):
           for ix, file in enumerate(f[:]):#[:20000]
                           #print (file)
                      if ".png" in file.lower(): #".jpg" or 
                          #img = cv2.imread(os.path.join(r, file))
                          self.file[file.split(".")[0]] = [os.path.join(r, file)]
                      if ".jpg" in file.lower(): #".jpg" or 
                          #img = cv2.imread(os.path.join(r, file))
                          #img = cv2.resize(img, (28,28))
                          self.file[file.split(".")[0]] = [os.path.join(r, file)]   
                      if ".jpeg" in file.lower(): #".jpg" or 
                          #img = cv2.imread(os.path.join(r, file))
                          #img = cv2.resize(img, (28,28))
                          self.file[file.split(".")[0]] = [os.path.join(r, file)]                            
                           
                      if ".json" in file.lower():
                          #jsf = open(os.path.join(r, file), 'r')
                          self.label[file.split(".")[0]] = [os.path.join(r, file)] 

    
def copy_dict_char():    
    IMG = DATA()
    IMG.parseIMG("CHAR_CUT_RU")
    IMG.parseIMG("CHAR_CUT_UA")

    print (len(IMG.file))

    H = {}

    for p in IMG.file:
        idx = p.split("_")[-1]
        try:
            H[idx].append(IMG.file[p][0])
        except KeyError:
            H[idx] = [IMG.file[p][0]]
        #print (p, IMG.file[p][0])
    for p in H:
        g_name = f"CHAR_SORT/{p}"
        os.mkdir(g_name)
        for m in H[p]:
            img = cv2.imread(m)
            file_name = m.split("/")[-1]
            cv2.imwrite(g_name+f"/{file_name}", img)
        print (p, len(H[p]))
        

    print (len(H))


#PARSING CHAR_CUT_RU/
#PARSING CHAR_CUT_UA/
#99329
#3 7599
#4 5205
#9 5548
#X 2406
#B 7135
#H 2890
#5 7991
#6 4864
#A 7122
#0 5638
#M 2418
#1 5783
#7 4992
#2 4579
#T 3086
#P 2032
#C 4137
#8 4826
#E 2901
#O 2895
#K 2928
#Y 913
#I 1441
#23
def imgs(x):
    cv2.imshow('Rotat', np.array(x))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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


#def gen_plate_with_cut():
#BG = DATA()
#BG.parseIMG("/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/PLAYGROUND/OCR/GAN/bgs")


#IMG = DATA()
#IMG.parseIMG("CHAR_SORT")


#for M in range(100000):
#    L = random.choice([1,2,3,4,5,6,7,8,9])
#    print(len(IMG.file))
#    ls = []
#    ls_mask = []
#    size_w = []
#    size_h = [] 
#    char = ""

#    for i in range(L):
#        A = random.choice(list(IMG.file.keys()))
#        char += A.split("_")[-1]
#        #img = cv2.imread(IMG.file[A][0], cv2.IMREAD_GRAYSCALE) / 255.
#        img = cv2.imread(IMG.file[A][0])
#        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#        lower = np.array([50, 10, 10])
#        upper = np.array([120, 255, 255])
#        mask = cv2.inRange(hsv, lower, upper)
#        res = cv2.bitwise_and(img, img, mask=mask)        
#        imgs(img)
#        for i in range(img.shape[0]):
#            for j in range(img.shape[1]):
#                print (sum(img[i,j])/3)
#                A = sum(img[i,j])/3
#                if A < 90.0:
#                    img[i,j] = [0,0,0]
#                if A > 90.0:
#                    img[i,j] = [255,255,255]
#        
#        imgs(img)
#-------------------------------------------_>        
D = DATA()
D.parseIMG("GEN_data_v1")

FONT_HEIGHT = 32  # Pixel size to which the chars are resized

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



        
