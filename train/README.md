https://ru.wikipedia.org/wiki/Европейские_регистрационные_знаки_транспортных_средств </br>
https://www.avtobeginner.ru/nomer/ </br>
https://github.com/shoorick/russian-road-sign-font </br>
https://stackoverflow.com/questions/43060479/how-to-get-the-font-pixel-height-using-pils-imagefont-class </br>
https://itnext.io/how-to-wrap-text-on-image-using-python-8f569860f89e </br>
https://github.com/python-pillow/Pillow/issues/3977 </br>
http://design-mania.ru/downloads/fonts/raspoznavanie-shriftov/ </br>

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
