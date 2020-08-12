https://ru.wikipedia.org/wiki/Европейские_регистрационные_знаки_транспортных_средств </br>
https://ru.wikipedia.org/wiki/Регистрационные_знаки_транспортных_средств_в_России </br>
https://www.google.com/search?channel=fs&sxsrf=ALeKk03-9wmjH5WdytmWBvWY-BRP3Man_w:1597146493374&source=univ&tbm=isch&q=%D0%B7%D0%BD%D0%B0%D0%BA%D0%B8+%D0%B3%D0%BE%D1%81%D1%83%D0%B4%D0%B0%D1%80%D1%81%D1%82%D0%B2%D0%B5%D0%BD%D0%BD%D1%8B%D0%B5+%D1%80%D0%B5%D0%B3%D0%B8%D1%81%D1%82%D1%80%D0%B0%D1%86%D0%B8%D0%BE%D0%BD%D0%BD%D1%8B%D0%B5+%D1%82%D1%80%D0%B0%D0%BD%D1%81%D0%BF%D0%BE%D1%80%D1%82%D0%BD%D1%8B%D1%85+%D1%81%D1%80%D0%B5%D0%B4%D1%81%D1%82%D0%B2+%D1%88%D1%80%D0%B8%D1%84%D1%82&client=ubuntu&sa=X&ved=2ahUKEwiJ2JCripPrAhXosYsKHbDPCEYQsAR6BAgKEAE </br>
https://www.avtobeginner.ru/nomer/ </br>
https://github.com/shoorick/russian-road-sign-font </br>
https://stackoverflow.com/questions/43060479/how-to-get-the-font-pixel-height-using-pils-imagefont-class </br>
https://itnext.io/how-to-wrap-text-on-image-using-python-8f569860f89e </br>
https://github.com/python-pillow/Pillow/issues/3977 </br>
http://design-mania.ru/downloads/fonts/raspoznavanie-shriftov/ </br>
http://docs.cntd.ru/document/1200160380 </br>
https://www.car72.ru/nomer/ </br>

Exact size depends on many factors. I'll just show you how to calculate different metrics of font.
<img src="https://i.stack.imgur.com/gSBad.png" alt="">
font = ImageFont.truetype('arial.ttf', font_size)
ascent, descent = font.getmetrics()
(width, baseline), (offset_x, offset_y) = font.font.getsize(text)

    Height of red area: offset_y
    Height of green area: ascent - offset_y
    Height of blue area: descent
    Black rectangle: font.getmask(text).getbbox()

Hope it helps.

малайзия, индонезия, вьетнам, филипины, беларусь, израиль, сингапур </br>
Использую mongo db.</br>
Структура состоит из 3 коллекций(таблиц)</br>
1 все что пришло на сервер</br>
2 ответы которые алгоритм считает правильными</br>
3 неправильные</br>
Структура столбцов коллекции:</br>
1 файл - пока не знаю хранить base64, или ссылку на изображение в файловой системе</br>
2 Ответ детектора:</br>
a) координаты</br>
б) коэффициент активации (уверенность сети в своем ответе)</br>
3 Ответ OCR</br>
а) ответ</br>
б) коэффициент для каждой буквы или для общего текста</br>
Изображения буду хранить в исходном состоянии</br>
