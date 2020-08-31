import requests

acc_key = "1314995842:AAFIp92pZYhSpGhaeX811fGD63-KazTbiu8"
#t.me/deep_anpr_bot
files = {'certificate': open('ssl/cert.crt', 'rb')}
my_l = "https://api.telegram.org/bot"+acc_key+"/setWebhook?url=https://95.216.240.243:8443"
r = requests.post(my_l, files=files)
print (r.text)
my_l = "https://api.telegram.org/bot"+acc_key+"/getWebhookInfo"
r = requests.get(my_l)
print (r.text)
