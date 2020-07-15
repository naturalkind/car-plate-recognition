# -*- coding: utf-8 -*-
import numpy


DIGITS = "0123456789"
#А, В, Е, К, М, Н, О, Р, С, Т, У, Х
#LETTERS = "АВЕКМНОРСТУХ"
#LETTERS = "ABEKMHOPCTYX"

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARS = LETTERS + DIGITS + " "
CHARS1 = LETTERS + DIGITS

def softmax(a):
    exps = numpy.exp(a.astype(numpy.float64))
    return exps / numpy.sum(exps, axis=-1)[:, numpy.newaxis]

def sigmoid(a):
  return 1. / (1. + numpy.exp(-a))

