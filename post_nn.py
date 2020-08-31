# -*- coding: utf-8 -*-
import numpy as np
import threading, cv2, os, time, requests, json, base64
from tornado.escape import json_encode
from bson.objectid import ObjectId



size = 416


def imgs(x):
      cv2.imshow('Rotat', np.array(x))
      cv2.waitKey(0)
      cv2.destroyAllWindows()

class DATA(object):
        def __init__(self):
            #self.file = []
            self.file = {}
            self.json = []
        def parseIMG(self, dir_name):
                path = dir_name+"/"
                print ("PARSING",path)
                for r, d, f in os.walk(path):
                    for ix, file in enumerate(f):
                        #print (file)
                        if ".jpeg" in file:
                           #self.file[file.split(".")[0]] = [os.path.join(r, file)]
                           #self.file.append(os.path.join(r, file))
                           self.file[file] = os.path.join(r, file)
                        if ".json" in file:
                           #self.file[file.split(".")[0]] = [os.path.join(r, file)]
                           self.json.append(os.path.join(r, file))   




class record():
     def __init__(self,idx):
         self.file = open(str(idx)+'_error.txt', 'w')
     def save(self, i):
         #print i
         self.file.write(i+"\n") 


def Post(data):
    url = "http://95.216.240.243:8800/"
    #_, data = cv2.imencode('.jpg', data) #base64.b64encode(data.encode())#str(data.encode("base64"))
    #data = base64.b64encode(data).decode("utf-8")
    #print (base64.b64decode(data), type(data))

    payload = json.dumps({"image":base64.b64encode(data).decode()})#base64.b64decode(data)
    #print (payload)
    headers = { 'Content-Type': "application/json",
                'cache-control': "no-cache" }
    response = requests.request("POST", url, data=payload, headers=headers)
    return response.text#json.loads()



if __name__ == "__main__":
    D = DATA()
    D.parseIMG("train")
    for o in D.file:
       #print (o)
       F = open(D.file[o],"rb").read()
       answ = Post(F)
       print (">>>")
       #imgs()
       
