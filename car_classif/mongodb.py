# -*- coding: utf-8 -*-
import pymongo
import os
import gridfs
import cv2
import numpy as np
import uuid
from bson.objectid import ObjectId

def imgs(x):
      cv2.imshow('Rotat', np.array(x))
      cv2.waitKey(0)
      cv2.destroyAllWindows()

mongodb_uri = os.getenv('MONGODB_URI', default='mongodb://localhost:27017/')

class DataBase(object):
     def __init__(self, test="test"):
         self.client = pymongo.MongoClient(mongodb_uri) #MongoClient('localhost', 27017)

         self.db = self.client[test]
         self.file = gridfs.GridFS(self.db)
         self.n ="collection" 
       
     def create_collection(self, x): #self.n
                self.n = x

     def create_post(self, post):
                 post_id = self.db[self.n].insert_one(post).inserted_id
                 return post_id

     def create_many(self, post):
                 self.db[self.n]
                 post_id = self.db[self.n].insert_many(post)
                 return post_id        

     def see_all_post(self):
         list = []
         for i in self.db[self.n].find():
              i["_id"] = str(i["_id"])
              list.append(i)
         return list
         
     def see_all_post_v1(self, x, y):
         return list(self.db[self.n].find().skip(x).limit(y))
         
     def see_all_post_v2(self, x, y, z):
         if z == None:
            return list(self.db[self.n].find().skip(x).limit(y))
         else:
            return list(self.db[self.n].find(z).skip(x).limit(y))
         
     def find_by_id(self, idx):
         return list(self.db[self.n].find(idx))
         
     def find_many_post(self, idx):
         #print (idx, self.n)
         return list(self.db[self.n].find(idx))         
 
     def see_post(self, idx):
         return self.db[self.n].find_one(idx)
         
     def upd_post(self, idx, post):
         return self.db[self.n].update_one({"_id" : idx},
                                           {"$set": post}, upsert=True)

     def del_all_post(self):
         self.db[self.n].delete_many({}) 

     def del_post(self):
         for i in self.db[self.n].find():
                 result = self.db[self.n].find_one(i["_id"])
                 result = self.db[self.n].delete_one(result)
                 result.deleted_count 

     def del_one_post(self, idx):
                 result = self.db[self.n].find_one({"_id":idx})
                 result = self.db[self.n].delete_one(result)
                 result.deleted_count 
                 
     # Список баз
     #кол-во всех данных
     def count(self):
         return self.db[self.n].find().count()
     #кол-во фильтрованных данных
     def count1(self, x):
         return self.db[self.n].find(x).count()    
     # Удалить базу
     def del_db_by_name(self, x):
         self.client.drop_database(x)
     # Удалить коллекцию
     def del_collection(self):
         result = self.db[self.n].drop() 

     def see_collection(self):
        list = []
        for i in self.db.collection_names(): #list_
           print(i)
           list.append(i)
        return list

     def see_client(self):
         for idb in self.client.database_names():
             print (idb, "Имя базы данных")
    
db = DataBase("drive2")
             
def images_data():
        img = cv2.imread("10700181-Image-1.jpeg")
        S = img.shape
        print (img.shape)
        _, img = cv2.imencode(".jpeg",img)
        img = img.tostring()
        #imgs(img)
        #db = DataBase("anpr")
        #imageID = db.file.put(img.tostring(), encoding='utf-8') # ложу файлы
        #print (imageID)#
        #------------------_>
        #print (db.db['fs.files'].find_one())
        #print (list(db.db['fs.files'].find()))
        #print (list(db.file.find()))#(db.file.list())
        #------------------_>
        #img = db.file.find_one({"_id":"5f480851b2c79a1ff0b884ca"})
        #img = db.db['fs.files'].get("5f480851b2c79a1ff0b884ca")
        #img = db.file.get(ObjectId("5f480851b2c79a1ff0b884ca")).read()
        #print (img)
        nparr = np.fromstring(img, np.uint8)
        #nparr = np.frombuffer(img, np.uint8)
        #img = np.reshape(nparr, S)
        #nparr = nparr.reshape(1, -1)
        img_t = cv2.imdecode(nparr, cv2.IMREAD_COLOR) #cv2.IMREAD_UNCHANGED
        print (nparr, img_t)
        imgs(img_t)
 
#def imgs_data_write():
#        db = DataBase("anpr")
#        img = cv2.imread("10700181-Image-1.jpeg")
#        imageID = db.file.put(img.tostring(), encoding='utf-8')
#        dump = {
#            'name': 'myTestSet',
#            'images': [
#                {
#                    'imageID': imageID,
#                    'shape': img.shape,
#                    'dtype': str(img.dtype)
#                }
#            ]
#        }   
#        db.create_collection("anpr") 
#        db.create_post(dump)
#def imgs_data_read():        
#        db = DataBase("anpr")
#        db.create_collection("anpr") 
#        for i in db.see_all_post():
#            print (i["images"][0]['imageID'], "\n")
#            
#            gOut = db.file.get(i["images"][0]['imageID']) 
#            img = np.frombuffer(gOut.read(), dtype=np.uint8)

#            img = np.reshape(img, i["images"][0]['shape'])
#            imgs(img)  

#db = DataBase("anpr")

def imgs_data_write(x, dump):
        name = str(uuid.uuid4())[:7]
        print(name)
        imageID = db.file.put(x.tostring(), encoding='utf-8')
#        dump = {
#            'name': name,
#            'images': imageID,
#            'answ': answ,
#            'answ_sw': answ_sw,
#            'score_yolo': score,
#            'coord': coord,
#            'coord_sw': coord_sw
#        }   
        dump['images'] = imageID
        db.create_collection("anpr") 
        print(db.create_post(dump))
#4c9f2f2
#5f4c07afb2c79a42dc5d3390
def imgs_data_read():
        #db = DataBase("anpr")
        db.create_collection("anpr") 
        for i in db.see_all_post():
               print (i)
            #if i["name"] == "name":
               img = db.file.get(i["images"]).read()
               nparr = np.fromstring(img, np.uint8)
               img_t = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
               imgs(img_t)
          
 
            
#https://coderoad.ru/49493493/Python-хранить-изображение-cv-в-mongodb-gridfs         
#https://dev-gang.ru/article/integracija-mongodb-s-python-s-ispolzovaniem-pymongo-9hmv4a77cw/            
#https://developer.mongodb.com/how-to/storing-large-objects-and-files
#https://stackoverflow.com/questions/49493493/python-store-cv-image-in-mongodb-gridfs
#from bs4 import BeautifulSoup
if __name__ == "__main__":
        print ("Start")
        #db = DataBase("anpr")
#--->        
        db.create_collection("anpr") 
        print (len(db.see_all_post()))
        #imgs_data_read()
        #db.del_all_post()
        #check
#        for i in db.see_all_post():
#               print (i)
#               img = db.file.get(i["images"]).read()
#               nparr = np.fromstring(img, np.uint8)
#               img_t = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#               im_name = 'check/'+i['name'] + ".jpg"
#               cv2.imwrite(im_name, img_t) 
#               file = open('check/'+i['name'] + ".txt","w")
#               file.write(i["answ"])
#               file.close()
               #print (img_t.shape)
               #imgs(img_t)
        #images_data()
        
#        col = db.see_collection()
#        for i in col:
#           db.create_collection(i)
#           db.del_collection()

        #t = db.see_collection()
        #db.create_collection("files")
        #print (db.see_all_post_v2(0,0, {"date":"27.08.2020 08:07:11"}))
        

        
        

