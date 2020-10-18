# -*- coding: utf-8 -*-
from fake_useragent import UserAgent
import tornado.httpserver
import tornado.ioloop
import tornado.web
import ssl
import json
import time
import requests as R
import numpy as np
import cv2
import random
import os
from GPUi5 import gg as get_small_images
from mongodb import *

class RequestLib(object):
    def __init__(self):
        self.session = R.session()
        self.session.proxies = {}
        self.headers = {}
        self.headers['User-agent'] = UserAgent().random
        self.headers['Accept-Language'] = "en,en-US;q=0,5"
        self.headers['Content-Type'] = "application/x-www-form-urlencoded"
        self.headers['Connection'] = "keep-alive"
        self.headers['Accept'] = "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"

    def get(self, http, proxy=False):
        get_page = self.session.get(http, headers=self.headers)#, timeout=(10, 10)) 
        return get_page
            
sess = RequestLib()
acc_key = "1314995842:AAFIp92pZYhSpGhaeX811fGD63-KazTbiu8"

class getToken(tornado.web.RequestHandler):
    def get(self):
        print ("GET")
        self.write("hello")
    def post(self):
        print ("POST")
        data = json.loads(self.request.body)
        #print (data)
        if "photo" in data["message"].keys():
           chat_id = data["message"]["from"]["id"]
           photo_id = data["message"]["photo"][-1]["file_id"]
           width_img = data["message"]["photo"][-1]["width"]
           height_img = data["message"]["photo"][-1]["height"]
           url = "https://api.telegram.org/bot"+acc_key+"/getFile?file_id="+ photo_id
           files = sess.get(url)
           data = json.loads(files.text)
           file_path = data["result"]["file_path"]
           files = sess.get("https://api.telegram.org/file/bot"+acc_key+"/"+file_path)
           img = files.content
           nparr = np.fromstring(img, np.uint8)
           img_t = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
           answer = get_small_images(img_t)
           if answer[1] != None:
                imgs_data_write(nparr, str(answer[2]), str(answer[-2]), str(answer[-3]), str(answer[1]), str(answer[-1]))
                print (img_t.shape, answer[2])
                ANSW = "{}; {}; coord 1: {}; coord 2: {}; score: {};".format(str(answer[2]), str(answer[-2]), 
                                                                             str(answer[1]), str(answer[-1]),
                                                                             str(answer[-3]))
                #+"sc;"++";"
                url = "https://api.telegram.org/bot"+acc_key+"/sendMessage?chat_id="+str(chat_id)+"&text="+ANSW+"&parse_mode=html"
                r = sess.get(url)
                url = "https://api.telegram.org/bot"+acc_key+"/sendPhoto";
                files = {'photo': answer[0]}
                data = {'chat_id' : chat_id}
                r = R.post(url, files=files, data=data) 
           if answer[1] == []:
                url = "https://api.telegram.org/bot"+acc_key+"/sendMessage?chat_id="+str(chat_id)+"&text="+str("не распознано")+"&parse_mode=html"
                imgs_data_write(nparr, "", "", "", "", "")
                r = sess.get(url)

        #chat_id = data["message"]["from"]["id"]
        #text_mess = data["message"]["text"]
        #print ("POST", chat_id, data)
#        if text_mess == "parser":
#           print ("PARSER")
        #self.redirect("https://api.telegram.org/bot"+acc_key+"/sendMessage?chat_id="+chat_ID+"&text="+op+"&parse_mode=html")
        #self.write(op)
        #files = sess.get("https://api.telegram.org/bot"+acc_key+"/getFile?file_id=AgACAgIAAxkBAANYX0TtSf9F29qtohwd4fRt3tbPUOwAAm6uMRs9VyhK-m0Xn2NXE9LNBNmWLgADAQADAgADeAADkXkAAhsE")
        #files = sess.get("https://api.telegram.org/file/bot"+acc_key+"/photos/file_1.jpg")
      
        #with open("file_1.jpg", 'wb') as new_file:
        #    new_file.write(files.content)

application = tornado.web.Application([
    (r'/', getToken),
])

if __name__ == '__main__':
    http_server = tornado.httpserver.HTTPServer(application, ssl_options={"certfile":"ssl/cert.crt",
                                                                          "keyfile":"ssl/cert.key",
                                                                          "ssl_version": ssl.PROTOCOL_TLSv1})
    #http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8443)
    tornado.ioloop.IOLoop.instance().start()
