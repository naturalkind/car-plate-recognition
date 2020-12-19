from __future__ import print_function, division

from keras.layers import LeakyReLU
from keras.layers import UpSampling2D, Conv2D
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D
from keras.layers import Reshape, Lambda
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import layers
from keras.models import model_from_json
import tensorflow as tf

import matplotlib.pyplot as plt

import sys, cv2, os
import json

import numpy as np

from GPUi6 import gg 
#from gen_all_1 import *
#import gen_all_1
#from gen_rus_plate4 import GenPlate
#from gen_rus_plate4 import DATA

#Test.parseLAB("test_gen")
###########################
CHARS = "ABEKMHOPCTYX" + "0123456789" + " "
####################
################
#print (Test.file.keys())
def get_data(plate_rus, D):
        def code_to_vec(code):
            while len(code) < 9:
               code += " "
            #print (len(code))
            def char_to_vec(c):
                y = np.zeros((len(CHARS),))
                #print c
                y[CHARS.index(c)] = 1.0
                return y

            c = np.vstack([char_to_vec(c) for c in code])
            #print (c.shape)
            return c.flatten()
        
        arr_rus = []
        arr_rus_lab = []
        for j in plate_rus:
            imgh = cv2.imread(D.file[j][0]).astype(np.float32) / 255.
            imgh = cv2.resize(imgh, (128, 64))
            #imgs_v(imgh[:,:,0])
            #print (imgh.shape)
            try:
                json_open = json.load(open(D.label[j][0], "r"))
                vec_arr = code_to_vec(json_open["description"])
                #print (vec_arr, np.reshape(vec_arr,(9,22)))
            except KeyError:
                vec_arr = code_to_vec(j.split("_")[-1])
                #print (j.split("_")[-1])
            arr_rus_lab.append(vec_arr)
            arr_rus.append(imgh[:,:,:1])
        return np.array(arr_rus), np.array(arr_rus_lab)

def imgs_v(x):
      cv2.imshow('Rotat', np.array(x))#, dtype=np.uint8
      #cv2.waitKey(1)
      
      cv2.waitKey(0)
      cv2.destroyAllWindows()

def build_cnn_classif():

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(64,128,1), padding="valid"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="valid"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        #model.add(Dense(1, activation='sigmoid'))

        model.add(Dense(9*len(CHARS), activation='sigmoid')) #'softmax'
#	# Compile model
#	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#	return model

        model.summary()

        img = Input(shape=(64,128,1))
        validity = model(img)

        return Model(img, validity)

#Сделать шум от 0 до 36 
#для вектора 1x100 
def get_data(plate_rus):
        def code_to_vec(code):
            while len(code) < 9:
               code += " "
            #print (len(code))
            def char_to_vec(c):
                y = np.zeros((len(CHARS),))
                #print c
                y[CHARS.index(c)] = 1.0
                return y

            c = np.vstack([char_to_vec(c) for c in code])
            #print (c.shape)
            return c.flatten()
        
        arr_rus = []
        arr_rus_lab = []
        for j in plate_rus:
            imgh = cv2.imread(gen_all_1.H4.file[j][0]).astype(np.float32) / 255.
            imgh = cv2.resize(imgh, (128, 64))
            #imgs_v(imgh[:,:,0])
            #print (imgh.shape)
            json_open = json.load(open(gen_all_1.H4.label[j][0], "r"))
            vec_arr = code_to_vec(json_open["description"])
            #print (vec_arr, np.reshape(vec_arr,(9,22)))
            arr_rus_lab.append(vec_arr)
            arr_rus.append(imgh[:,:,:1])
        return np.array(arr_rus), np.array(arr_rus_lab)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 64
        self.img_cols = 128
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 192#40#36#100

        optimizer = Adam(0.0002, 0.5)
        #----------------->
        # Classific
        #----------------->
        self.model_res = build_cnn_classif()
        try:
            listOfNumpyArrays = np.load('class.npy', allow_pickle=True)
            self.model_res.set_weights(listOfNumpyArrays)
        except:
            print ("Not load")
        self.model_res.compile(loss='categorical_crossentropy',
                               optimizer=optimizer,
                               metrics=['accuracy'])

        #----------------->
        # GAN 
        #----------------->
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        try:
            listOfNumpyArrays = np.load('generator.npy', allow_pickle=True)
            self.generator.set_weights(listOfNumpyArrays)
        except:
            print ("Not load")

        #model.layers[i].set_weights(listOfNumpyArrays)    
        #model.get_layer(layerName).set_weights(...)
        #model.set_weights(listOfNumpyArrays)
        # load json and create model
        #json_file = open('model.json', 'r')
        #loaded_model_json = json_file.read()
        #json_file.close()
        #self.generator = model_from_json(loaded_model_json)
        # load weights into new model
        #self.generator.load_weights("model.h5")
        #print("Loaded model from disk")

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 8 * 8, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((8, 8, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D(size=(1, 2)))
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def detect1(self, name):
        imgh_orig = cv2.imread(name).astype(np.float32) / 255.
        imgh = cv2.resize(imgh_orig, (128, 64))

        #imgh = gg(imgh_orig)
        aaa = self.model_res.predict(np.reshape(imgh[:,:,:1], (1, 64, 128, 1)))
        for k in range(aaa.shape[0]):
           best = np.argmax(np.reshape(aaa[k], [-1, 9, len(CHARS)]), 2)
           print ("".join(CHARS[int(i)] for i in best[0]))
                        # serialize model to JSON
           print ("------------------------------>")
           imgs_v(imgh_orig)

    def detect(self, name):
        imgh_orig = cv2.imread(name)#.astype(np.float32) / 255.
        #imgh = cv2.resize(imgh_orig, (128, 64))

        imgh = gg(imgh_orig)

        aaa = self.model_res.predict(imgh)
        for k in range(aaa.shape[0]):
           best = np.argmax(np.reshape(aaa[k], [-1, 9, len(CHARS)]), 2)
           print ("".join(CHARS[int(i)] for i in best[0]))
                        # serialize model to JSON
           print ("------------------------------>", imgh.shape)
           imgs_v(imgh[0,:,:,:])


if __name__ == '__main__':
    dcgan = DCGAN()
    #dcgan.detect1("/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/PLAYGROUND/OCR/GAN/autoriaNumberplateOcrRu-2019-08-30/img/11_6_2014_19_4_42_41_0.png")
    dcgan.detect("/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/PLAYGROUND/build_parse_drive2/_data_img/lada/2111/1284636/437d916s-960.jpg")
    
