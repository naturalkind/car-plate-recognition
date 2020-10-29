### Видео изготовления:</br>
https://www.youtube.com/watch?v=TPfbKqNIYFY </br>
https://ru.wikipedia.org/wiki/Европейские_регистрационные_знаки_транспортных_средств </br>
https://ru.wikipedia.org/wiki/Регистрационные_знаки_транспортных_средств_в_России </br>
https://www.google.com/search?channel=fs&sxsrf=ALeKk03-9wmjH5WdytmWBvWY-BRP3Man_w:1597146493374&source=univ&tbm=isch&q=знаки+государственные+регистрационные+транспортных+средств+шрифт&client=ubuntu&sa=X&ved=2ahUKEwiJ2JCripPrAhXosYsKHbDPCEYQsAR6BAgKEAE </br>
https://www.avtobeginner.ru/nomer/ </br>
https://github.com/shoorick/russian-road-sign-font </br>
https://stackoverflow.com/questions/43060479/how-to-get-the-font-pixel-height-using-pils-imagefont-class </br>
https://itnext.io/how-to-wrap-text-on-image-using-python-8f569860f89e </br>
https://github.com/python-pillow/Pillow/issues/3977 </br>
http://design-mania.ru/downloads/fonts/raspoznavanie-shriftov/ </br>
http://docs.cntd.ru/document/1200160380 </br>
https://www.car72.ru/nomer/ </br>
https://habr.com/ru/company/smartengines/blog/264677/ </br>
https://habr.com/ru/post/439330/</br>
https://nanonets.com/blog/ocr-with-tesseract/ </br>
https://www.learnopencv.com/deep-learning-based-text-recognition-ocr-using-tesseract-and-opencv/ </br>

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
https://github.com/fengxinjie/Transformer-OCR </br>
https://github.com/soumik12345/Automatic-Number-Plate-Recognition </br>
https://github.com/watsonyanghx/CNN_LSTM_CTC_Tensorflow </br>
https://github.com/synckey/tensorflow_lstm_ctc_ocr </br>
https://github.com/weinman/cnn_lstm_ctc_ocr </br>
https://github.com/wushilian/STN_CNN_LSTM_CTC_TensorFlow/ </br>
https://github.com/leitro/handwrittenDigitSequenceRec </br>
https://github.com/jimmysh2/CNN-LSTM-CTC-OCR-Tensorflow </br>
https://github.com/zfxxfeng/cnn_lstm_ctc_ocr_for_ICPR </br>
https://github.com/githubharald/SimpleHTR </br>
https://github.com/bai-shang/crnn_ctc_ocr_tf </br>
https://arxiv.org/pdf/1910.05085.pdf </br>
https://cs231n.github.io/convolutional-networks/#conv </br>
https://github.com/ilovin/lstm_ctc_ocr </br>
https://github.com/quangnhat185/Plate_detect_and_recognize </br>
https://github.com/sergiomsilva/alpr-unconstrained </br>
https://github.com/matthewearl/deep-anpr </br>
