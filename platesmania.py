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
                        if "txt" == G:
                           self.txt[file.split(".")[0]] = [os.path.join(r, file)]
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
#--------------------------------------------->
class RequestLib():
    def __init__(self):
        self.session = R.session()
        self.session.proxies = {}
        #self.session.proxies['http'] = 'socks5://127.0.0.1:9050'
        #self.session.proxies['https'] = 'socks5://127.0.0.1:9050'
        #self.session.proxies['http'] = "http://194.156.118.188:58264@gsVYQSORRr:6orWkDjr3c"
        #self.session.proxies['https'] = "http://194.156.118.188:58264@gsVYQSORRr:6orWkDjr3c"  
        self.headers = {}
        #self.headers['host'] = "img03.platesmania.com" 
        self.headers['User-agent'] = UserAgent().random
        self.headers['Accept-Language'] = "en,en-US;q=0,5"
        self.headers['Content-Type'] = "application/x-www-form-urlencoded"
        self.headers['Connection'] = "keep-alive"
        self.headers['Accept'] = "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
        self.headers['Cookie'] = "__cfduid=d7fbd5a5faa3429f34b5c5dd01af3308d1601054241; __gads=ID=7de32003c67b6d9a:T=1601054244:S=ALNI_MZq382Lye1kk32_xyTpS_nbgMd5Yg; cto_bundle=P-TPBF94UHh5aU1TQVQ4VjcwY0lIU0taaTBFc3U4Z2xtMG40NEtFeG5hT1dHbXZVdGQwbkdmQWFmMG1HYWNzVkJic1luSGVvUjVNMkdHdFVBVllwWmE2NkxLYWxaVXo5OWl4M1hXMGtHJTJGMjYxWEtsMm9QOVQ1bHlIeXVKTU5jbSUyQjBEMUNkTzRBVjhwRDNxTHNsUiUyQkxpdlU1UGclM0QlM0Q; _ym_uid=1596453110626116202; _ym_d=1601054253"
    def get(self, http):
        #print (http)
        get_page = self.session.get(http, headers=self.headers)#, timeout=(10, 10)) 
        return get_page#.text 
        
sess = RequestLib()
#212.60.7.67:58299:gsVYQSORRr:6orWkDjr3c
##<h1 class="pull-left">м003ас23</h1>
##<div class="panel-body">

#mod = ""
#for OPS in range(700000):
#    #l = f"http://88.99.70.88/ru/gallery{mod}"
#    #l = f"http://88.99.70.88/ua/gallery{mod}"
#    #l = f"http://88.99.70.88/by/gallery{mod}"
#    l = f"http://88.99.70.88/kz/gallery{mod}"
#    html = sess.get(l)
#    soup = BeautifulSoup(html, features = "html.parser")
#    J = soup.find_all('div', {"class": "col-sm-6"})
#    OPS+=1
#    mod = f"-{OPS}"
#    print (len(J), l)
#    for iJ in J:
#        B = iJ.find('div', {"class": "panel-body"})
#        #small_img = iJ.find('div', {"class": "col-xs-offset-3"})
#        try:
#                link = B.find('a')["href"]
#                small_img = B.find_all('img', {"class": "img-responsive"})
#                big_img = small_img[0]["src"].split("/")[-1].split(".")[0]
#                code = small_img[-1]["alt"].replace(" ", "")
#                small_img =small_img[-1]["src"]
#                foro_clear = f"http://88.99.70.88/ru/foto{big_img}"
#                #print (small_img[-1]["alt"].replace(" ", ""), small_img[-1]["src"], foro_clear) 
#                UID = str(uuid.uuid4())[:12] 
#                
#                filename  = f"platesmania_data_kz/{UID}_{code}.txt"
#                #print (filename, foro_clear, link)
#                file_txt = open(filename, "w")
#                tstr = f"{foro_clear}\n{small_img}\n{code}\n{link}\n"
#                file_txt.write(tstr)
#                file_txt.close()
#               
#        except:
#                pass
    #time.sleep(1)
#---------------------------------------------------->
D = DATA()
D.parseIMG("platesmania_data") 
#print (len(D.txt))
dict_data = {}
dict_data2 = {}
for i in D.txt:
    f = open(D.txt[i][0],"r").readlines()
    f = [i.split("\n")[0] for i in f]
    #dict_data[f[-1]] = [f[-2].replace("-",""), f[1], f[0].split("/")[-1]]
    print (f"http://img03.platesmania.com/{f[1].split('/')[-3]}/o/{f[1].split('/')[-1][:8]}.jpg")
    #print ([f[-2].replace("-",""), f[1], f"http://88.99.70.88/{f[-1].split('/')[1]}/{f[0].split('/')[-1]}"])
    response = sess.get(f"http://img03.platesmania.com/{f[1].split('/')[-3]}/o/{f[1].split('/')[-1][:8]}.jpg")
    #104.27.135.96
    print (response)
    if response.status_code == 200:
       with open(f'palatesmania_img/{f[-2].replace("-","")}.jpg', 'wb') as fli:
                 fli.write(response.content)
    time.sleep(random.choice([1,0.5,0.7,1.5,1.3,2]))             
print (len(dict_data.keys()), len(dict_data2.keys()))    

