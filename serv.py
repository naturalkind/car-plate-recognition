# -*- coding: utf-8 -*-
import tornado.ioloop
import tornado.web
import json
import re
import base64
import uuid
import gc
import cv2
import os, sys
import glob
import numpy
import time
import io
from GPUi5 import gg as get_small_images
from tornado.escape import json_encode
from mongodb import *
from serv_t import *


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index3.html", title="Нейронная сеть/Тренировка")
    def post(self):
        nameFile = str(uuid.uuid4())[:12]
        self.set_header("Content-Type", "application/json")
        start_time = time.time()
        data_json = json.loads(self.request.body)
        on = 'on'#data_json['on']
        if str(on) == 'on':
                start = time.time()
                file_bytes = numpy.asarray(bytearray(io.BytesIO(base64.b64decode(data_json['image'])).read()), dtype=numpy.uint8)
                nameFile = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                #print ("CLASS:", tp)
                
                answer = get_small_images(nameFile) 
                end = time.time()
                print ((end - start)/60, answer[1], answer[2])
                #dtimg = ocr.ocr_img(answer[0])
                  
                if answer[1] != None:
                   
                   obj = {"id": "", "text": str(answer[2]), "solved": True, "status": "OK", "type": "plate", "time": (time.time() - start_time), 'img': base64.b64encode(answer[0]).decode()}
                   imgs_data_write(file_bytes, str(answer[2]), str(answer[-1]), str(answer[1]))
                else:
                   obj = {"id": "", "text": "", "solved": False, "status": "OK", "type": "plate", "time": (time.time() - start_time), 'img': base64.b64encode(answer[0])}
                   imgs_data_write(file_bytes, "", "", "")

                op = json_encode(obj)
                self.write(op)


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
	(r"/(robots-AI.jpg)", tornado.web.StaticFileHandler, {'path':'./'}),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8800) #95.216.240.243
    http_server = tornado.httpserver.HTTPServer(application, ssl_options={"certfile":"ssl/cert.crt",
                                                                          "keyfile":"ssl/cert.key",
                                                                          "ssl_version": ssl.PROTOCOL_TLSv1})
    #http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8443)
    tornado.ioloop.IOLoop.current().start()


