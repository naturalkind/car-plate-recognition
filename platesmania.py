from stem import Signal
from stem.control import Controller
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import numpy as np
import requests as R
import time
import random
import json
import os
import ast
from bson.objectid import ObjectId
import uuid


#def my_proxy(PROXY_HOST,PROXY_PORT):
#    fp = webdriver.FirefoxProfile()
#    fp.set_preference("network.proxy.type", 1)
#    fp.set_preference("network.proxy.socks",PROXY_HOST)
#    fp.set_preference("network.proxy.socks_port",int(PROXY_PORT))

#    fp.update_preferences()
#    options = Options()
#    #options.add_argument('headless')
#    #options.add_argument("--headless")
#    #options.headless = True
#    return webdriver.Firefox(options=options, firefox_profile=fp)
##88.99.70.88
#proxy = my_proxy("127.0.0.1", 9050)
#proxy.get("http://88.99.70.88/")#("https://platesmania.com/ru/gallery")
##http://avto-nomer.ru/ru/gallery #https://myip.ru/
#html = proxy.page_source
#soup = BeautifulSoup(html, features = "html.parser")
#J = soup.find_all('div', {"class": "col-sm-6 col-xs-12"})
#print (J)

class RequestLib():
    def __init__(self):
        self.session = R.session()
        self.session.proxies = {}
        self.session.proxies['http'] = 'socks5://127.0.0.1:9050'
        self.session.proxies['https'] = 'socks5://127.0.0.1:9050'
        self.headers = {}
        self.headers['host'] = "avto-nomer.ru" 
        self.headers['User-agent'] = UserAgent().random
        self.headers['Accept-Language'] = "en,en-US;q=0,5"
        self.headers['Content-Type'] = "application/x-www-form-urlencoded"
        self.headers['Connection'] = "keep-alive"
        self.headers['Accept'] = "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    def get(self, http):
        #print (http)
        get_page = self.session.get(http, headers=self.headers)#, timeout=(10, 10)) 
        return get_page.text 
        
sess = RequestLib()
#<h1 class="pull-left">м003ас23</h1>
#<div class="panel-body">

mod = ""
for OPS in range(700000):
    #l = f"http://88.99.70.88/ru/gallery{mod}"
    #l = f"http://88.99.70.88/ua/gallery{mod}"
    #l = f"http://88.99.70.88/by/gallery{mod}"
    l = f"http://88.99.70.88/kz/gallery{mod}"
    html = sess.get(l)
    soup = BeautifulSoup(html, features = "html.parser")
    J = soup.find_all('div', {"class": "col-sm-6"})
    OPS+=1
    mod = f"-{OPS}"
    print (len(J), l)
    for iJ in J:
        B = iJ.find('div', {"class": "panel-body"})
        #small_img = iJ.find('div', {"class": "col-xs-offset-3"})
        try:
                link = B.find('a')["href"]
                small_img = B.find_all('img', {"class": "img-responsive"})
                big_img = small_img[0]["src"].split("/")[-1].split(".")[0]
                code = small_img[-1]["alt"].replace(" ", "")
                small_img =small_img[-1]["src"]
                foro_clear = f"http://88.99.70.88/ru/foto{big_img}"
                #print (small_img[-1]["alt"].replace(" ", ""), small_img[-1]["src"], foro_clear) 
                UID = str(uuid.uuid4())[:12] 
                
                filename  = f"platesmania_data_kz/{UID}_{code}.txt"
                #print (filename, foro_clear, link)
                file_txt = open(filename, "w")
                tstr = f"{foro_clear}\n{small_img}\n{code}\n{link}\n"
                file_txt.write(tstr)
                file_txt.close()
               
        except:
                pass
    #time.sleep(1)


