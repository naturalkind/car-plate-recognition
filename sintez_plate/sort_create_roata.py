from __future__ import print_function, division
import numpy as np
import sys, cv2, os
import json, time, uuid
import random
import uuid


#https://gautamnagrawal.medium.com/rotating-image-by-any-angle-shear-transformation-using-only-numpy-d28d16eb5076
#https://www.pythoninformer.com/python-libraries/numpy/image-transforms/
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
BG = DATA()
BG.parseIMG("/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/PLAYGROUND/OCR/GAN/bgs")


IMG = DATA()
IMG.parseIMG("CHAR_SORT")


for M in range(10000):
    L = random.choice([1,2,3,4,5,6,7,8,9])
    #print(len(IMG.file))
    ls = []
    ls_mask = []
    size_w = []
    size_h = [] 
    char = ""

    for i in range(L):
        A = random.choice(list(IMG.file.keys()))
        char += A.split("_")[-1]
        img = cv2.imread(IMG.file[A][0], cv2.IMREAD_GRAYSCALE) / 255.
        # Rotat

#-------------------------------------------_>        
        ANG = random.randint(-30,30)

        angle = ANG * (np.pi/180) #преобразование градусов в радианы
        cos_= np.cos(angle)
        sin_= np.sin(angle)
        tangent=np.tan(angle/2)
        
        height = img.shape[0] #определить высоту изображения
        width = img.shape[1] #определить ширину изображения
        
        new_height = round(abs(img.shape[0]*cos_)+abs(img.shape[1]*sin_))+1
        new_width = round(abs(img.shape[1]*cos_)+abs(img.shape[0]*sin_))+1

        img_zeros = np.zeros((new_height, new_width))
        img_zeros2 = np.zeros((new_height, new_width))
        
        original_centre_height = round(((img.shape[0]+1)/2)-1) # по отношению к исходному изображению
        original_centre_width = round(((img.shape[1]+1)/2)-1) # по отношению к исходному изображению
 
        new_centre_height= round(((new_height+1)/2)-1) # относительно нового изображения
        new_centre_width= round(((new_width+1)/2)-1) # относительно нового изображения    
        for i in range(height):
            for j in range(width):
                #координаты пикселя относительно центра исходного изображения
                y=img.shape[0]-1-i-original_centre_height                   
                x=img.shape[1]-1-j-original_centre_width                      

#                #координата пикселя относительно повернутого изображения
#                new_y=round(-x*sin_+y*cos_)
#                new_x=round(x*cos_+y*sin_)

                '''
                |1  -tan(𝜃/2) |  |1        0|  |1  -tan(𝜃/2) | 
                |0      1     |  |sin(𝜃)   1|  |0      1     |
                '''
                # shear 1
                new_x=round(x-y*tangent)
                new_y=y
                
                #shear 2
                new_y=round(new_x*sin_+new_y)      #since there is no change in new_x according to the shear matrix

                #shear 3
                new_x=round(new_x-new_y*tangent)              #since there is no change in new_y according to the shear matrix
                


                '''поскольку изображение будет повернуто, центр тоже изменится,
                   поэтому, чтобы приспособиться к этому, нам нужно будет изменить new_x и new_y по отношению к новому центру '''
                new_y = new_centre_height-new_y
                new_x = new_centre_width-new_x
                
#                img_zeros[new_y,new_x]=img[i,j] # запись пикселей в новое место назначения в выходном изображении
#                # добавление проверки для предотвращения ошибок при обработке
                if 0 <= new_x < new_width and 0 <= new_y < new_height and new_x>=0 and new_y>=0:
                    img_zeros[new_y,new_x]=1.0  # запись пикселей в новое место назначения в выходном изображении
                    img_zeros2[new_y,new_x]=img[i,j]
        #imgs(img_zeros)

#--------------------------------------_>        
        #
        ls.append(img_zeros)
        ls_mask.append(img_zeros2)
        
        size_h.append(img_zeros.shape[1])
        size_w.append(img_zeros.shape[0])

    #print (max(size_h), sum(size_w))  
    OUTPUT_SHAPE = [max(size_w), sum(size_h)]
    bg = generate_bg()
    #imgs(bg)
    #print (bg.shape)
    start_x = 0
    for ix, i in enumerate(ls):
        #imgs(i)
        LA = 1#random.choice([1,2,3])
        if LA == 1:
            temp = bg[0:size_w[ix], start_x:size_h[ix]+start_x]
            #bg[0:size_w[ix], start_x:size_h[ix]+start_x] = i[:,:]
            for HH in range(temp.shape[0]):
                for WW in range(temp.shape[1]):
                    if i[HH,WW] == 1.0:
                        temp[HH,WW] = ls_mask[ix][HH,WW]
            bg[0:size_w[ix], start_x:size_h[ix]+start_x] = temp
        elif LA == 2:
            bg[int((max(size_w)-size_w[ix])):size_w[ix]+int((max(size_w)-size_w[ix])), start_x:size_h[ix]+start_x] = i[:,:]
        else:
            bg[int((max(size_w)-size_w[ix])/2):size_w[ix]+int((max(size_w)-size_w[ix])/2), start_x:size_h[ix]+start_x] = i[:,:]
        #imgs(bg) # show one char to image
        start_x += size_h[ix]
        #print (i.shape, start_x, size_h[ix], (max(size_w)-size_w[ix])/2)
    #print (char)
    #imgs(bg)
    nameFile = str(uuid.uuid4())[:12]
    cv2.imwrite(f"GEN_data_v2/{nameFile}_{char}.jpg", bg*255.)

#BG = DATA()
#BG.parseIMG('/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/PLAYGROUND/OCR/train_non_gen/ocr_img')
#175707
#47848
#AAA = {}
#for o in BG.file:
#    #print (BG.label[o][0])
#    C = o.split("_")[-1]
#    try:
#        AAA[C].append(o)
#    except KeyError:
#        AAA[C] = [o]
#print (len(BG.file), len(AAA)) 
#for o in AAA:
#    if len(AAA[o])>1:
#        for b in AAA[o]:
#            
#            #print (len(AAA[o]))   
#            img = cv2.imread(BG.file[b][0])
##    print (o)
#            imgs(img)
#--------------------------------->
#AAA = {}
##181 319
def rename_file_autoria():
    BG.parseIMG('/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/PLAYGROUND/OCR/RAR')
    for o in BG.file:
        open_answ = json.loads(open(BG.label[o][0], "r").read())    
         
        nameFile = str(uuid.uuid4())[:12]
        
        print (o, BG.file[o][0])
        try:
            img = cv2.imread(BG.file[o][0])
            cv2.imwrite(f"/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/PLAYGROUND/OCR/RENAME/ALLDATA/{nameFile}_{open_answ['description']}.{BG.file[o][0].split('.')[-1]}", img)
        except:
            pass
    print (len(BG.file), len(AAA))
        
