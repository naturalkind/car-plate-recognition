import os
import cv2
import numpy as np
import json
import random
#import ocr_s
import uuid

def imgs(x):
      cv2.imshow('Rotat', np.array(x))
      cv2.waitKey(0)
      cv2.destroyAllWindows()

class DATA(object):
        def __init__(self):
            self.txt = {}
            self.file = {}
            
        def parseIMG(self, dir_name):
                path = "{}/".format(dir_name)
                print ("PARSING",path)
                for r, d, f in os.walk(path):
                    for ix, file in enumerate(f):
                        #print (file)
                        G = file.split(".")[-1]
                        if "jpg" == G:
                           self.file[file.split(".")[0]] = [os.path.join(r, file)]
                        if "png" == G:
                           self.file[file.split(".")[0]] = [os.path.join(r, file)]                           
                        if "json" == G:
                           self.txt[file.split(".")[0]] = [os.path.join(r, file)]
 
D = DATA()
D.parseIMG("data_non_rar") 
#print (len(D.file), len(D.txt))
temp_list = list(D.file)
random.shuffle(temp_list)
IX = 0
for i in temp_list:
    json_open = json.load(open(D.txt[i][0], "r"))
    #print (D.file[i][0])
    img = cv2.imread(D.file[i][0])
    UID = str(uuid.uuid4())[:12]  
    filename  = f"ocr_img/{UID}_{json_open['description']}.png"
    print (filename)
    cv2.imwrite(filename, img)


#for i in temp_list:
#    json_open = json.load(open(D.txt[i][0], "r"))
#    #print (D.file[i][0])
#    img = cv2.imread(D.file[i][0])
#    img_resized0 = cv2.resize(img, (128, 64))
#    img_ = img_resized0[:,:,0] / 255.
#    img_ = np.reshape(img_, [1,64,128])
#    ssss = ocr_s.modif_detect(img_)
#    #imgs(img)
#    if json_open["description"] == ssss:
#       IX += 1
#       print (json_open["description"], ssss, img_.shape, IX)
#       #imgs(img)
#    else:
#       #print (json_open["description"], ssss, i) 
#       UID = str(uuid.uuid4())[:12]  
#       filename  = f"ocr_img/{UID}_{json_open['description']}.png"
#       print (filename)
#       cv2.imwrite(filename, img)
print (len(D.file), len(D.txt), IX)       
