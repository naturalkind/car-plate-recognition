# -*- coding: utf-8 -*-
import json
import time
import requests
from datetime import date, timedelta, datetime
acc_key = "1314995842:AAFIp92pZYhSpGhaeX811fGD63-KazTbiu8"

#def get_chat(chat_ID):
#     params = {'chat_id': chat_ID}
#     my_l = "https://api.telegram.org/bot"+acc_key+"/getChat"           
#     r = requests.get(my_l, data=params).json()
#     print ("Ops, Error", r,  "\n") 

def get_chat(chat_ID):
     #params = {'peer': chat_ID}
     my_l = "https://api.telegram.org/bot"+acc_key+"/getChatHistory"           
     r = requests.get(my_l).json()
     print ("Ops, Error", r,  "\n") 

get_chat(1677356)
#{'update_id': 583012516, 'message': {'message_id': 855, 'from': {'id': 1677356, 'is_bot': False, 'first_name': 'Mash', 'username': 'breaking_mash', 'language_code': 'ru'}, 'chat': {'id': 1677356, 'first_name': 'Mash', 'username': 'breaking_mash', 'type': 'private'}, 'date': 1598891843, 'photo': [{'file_id': 'AgACAgIAAxkBAAIDV19NJ0N3vVrvuv_1Jilt98fr-ysrAAOvMRsKp2hK4OY2vxuAsKfOGQiSLgADAQADAgADbQADpOoFAAEbBA', 'file_unique_id': 'AQADzhkIki4AA6TqBQAB', 'file_size': 11030, 'width': 320, 'height': 126}, {'file_id': 'AgACAgIAAxkBAAIDV19NJ0N3vVrvuv_1Jilt98fr-ysrAAOvMRsKp2hK4OY2vxuAsKfOGQiSLgADAQADAgADeAADpuoFAAEbBA', 'file_unique_id': 'AQADzhkIki4AA6bqBQAB', 'file_size': 45388, 'width': 800, 'height': 316}, {'file_id': 'AgACAgIAAxkBAAIDV19NJ0N3vVrvuv_1Jilt98fr-ysrAAOvMRsKp2hK4OY2vxuAsKfOGQiSLgADAQADAgADeQADpeoFAAEbBA', 'file_unique_id': 'AQADzhkIki4AA6XqBQAB', 'file_size': 69504, 'width': 1280, 'height': 506}]}}

