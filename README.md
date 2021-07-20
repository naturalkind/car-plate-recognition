# распознавание автомобильных номеров 
### CNN(YOLOv3) -> CNN(VGG)
YOLOv3 https://cloud.mail.ru/public/eUEX/4cLSzTCbz 2 класса<br/>
VGG https://cloud.mail.ru/public/3ydY/4HEztvc2k OCR до 9 символов<br/>
python2/python3 serv.py<br/>
systemctl daemon:<br/>
#sudo nano /etc/systemd/system/bot.service<br/>
```
[Unit]
Description=ANPR

[Service]
Environment=LD_LIBRARY_PATH=/usr/local/cuda/lib64
Environment=/root/deepexperiment/car-plate-recognition
WorkingDirectory=/root/deepexperiment/car-plate-recognition
ExecStart=/usr/bin/python3 /root/deepexperiment/car-plate-recognition/serv.py
User=root
Restart=always
RestartSec=100
Type=simple

[Install]
WantedBy=multi-user.target
```
#sudo systemctl daemon-reload<br/>
#sudo systemctl enable bot.service<br/>
#sudo systemctl daemon-reload<br/>
#sudo systemctl start bot.service<br/>
#sudo systemctl status bot.service<br/>
#sudo systemctl stop bot.service<br/>

#sudo systemctl disable bot.service<br/>
#sudo systemctl restart bot.service<br/>

![Иллюстрация к проекту](/evilsadko/car-plate-recognition/blob/v0.1/github/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%20%D0%BE%D1%82%202020-08-23%2011-50-59.png?raw=true)
