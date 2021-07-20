# -*- coding: utf-8 -*-
import pymongo
import os
import gridfs
import cv2
import numpy as np
import uuid
import time
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
             
 
def imgs_data_write_v0():
        img = cv2.imread("10700181-Image-1.jpeg")
        imageID = db.file.put(img.tostring(), encoding='utf-8')
        dump = {
            'name': 'myTestSet',
            'images': [
                {
                    'imageID': imageID,
                    'shape': img.shape,
                    'dtype': str(img.dtype)
                }
            ]
        }   
        db.create_collection("anpr") 
        db.create_post(dump)
        
def imgs_data_read_v0():        
        db = DataBase("anpr")
        db.create_collection("anpr") 
        for i in db.see_all_post():
            gOut = db.file.get(i["images"][0]['imageID']) 
            img = np.frombuffer(gOut.read(), dtype=np.uint8)
            img = np.reshape(img, i["images"][0]['shape'])
            imgs(img)  


def imgs_data_write(x, answ, answ_sw, score, coord, coord_sw):
        name = str(uuid.uuid4())[:7]
        imageID = db.file.put(x.tostring(), encoding='utf-8')
        dump = {
            'name': name,
            'images': imageID,
            'answ': answ,
            'answ_sw': answ_sw,
            'score_yolo': score,
            'coord': coord,
            'coord_sw': coord_sw
        }   
        db.create_collection("anpr") 
        print(db.create_post(dump))
        
        
def imgs_data_read():
        db.create_collection("anpr") 
        for i in db.see_all_post():
               img = db.file.get(ObjectId(i["images"])).read()
               nparr = np.fromstring(img, np.uint8)
               img_t = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
               imgs(img_t)
               
          
 
            
print ("Start")
db = DataBase("anpr")


if __name__ == "__main__":
       
        #db.del_db_by_name("anpr") # Удалить базу данных
        #db.create_collection("anpr")
        #db.del_collection() # Удалить коллекцию
        db.see_collection()
        
        #imgs_data_write_v0()
        #imgs_data_read_v0()
        
        
        
        

        
        

