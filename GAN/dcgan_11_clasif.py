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


#from gen_all_1 import *
#import gen_all_1
import gen_rus_plate4 as Gen
#from gen_rus_plate4 import GenPlate
#from gen_rus_plate4 import DATA

Test = Gen.DATA()
Test.parseIMG("test_gen")
plate_rus_gen = Test.file.keys()
###########################
KL = Gen.GenPlate()
CHARS = KL.LETTER + KL.DIGIT + " "
####################
H4 = Gen.DATA()
H4.parseIMG("autoriaNumberplateOcrRu-2019-08-30")
plate_rus = H4.label.keys()
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
X_test, Y_test = get_data(plate_rus, H4)
X_train, Y_train = get_data(plate_rus_gen, Test)

def imgs_v(x):
      cv2.imshow('Rotat', np.array(x))#, dtype=np.uint8
      cv2.waitKey(1)
      
      #cv2.waitKey(0)
      #cv2.destroyAllWindows()

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
        self.model_res = self.build_cnn_classif()
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
        
    def build_cnn_classif(self):

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
        model.add(Dense(9*len(CHARS), activation='sigmoid')) #'softmax'

        model.summary()

        img = Input(shape=(64,128,1))
        validity = model(img)

        return Model(img, validity)



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

    def train(self, epochs, batch_size=30, save_interval=50):

        #plate_rus = gen_all_1.H4.label.keys()
#        x_train, y_train = get_data(plate_rus)
        #################

#        # Train
#        X_train, Y_train = x_train[100:], y_train[100:]
#        #X_train = X_train / 127.5 - 1.
#        # TEST
#        X_test, Y_test = x_train[:100], y_train[:100]



        #print (Y_train.shape, np.random.randint(0, len(plate_rus), batch_size), Y_train[0])
        #self.model_res
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        try:
            for epoch in range(epochs):
                for batch_idx in range(X_train.shape[0]//batch_size):
                    # Select a random half of images 
                    #idx = np.random.randint(0, len(plate_rus), batch_size)
                    #imgs = X_train[idx]
                    #labl = Y_train[idx]
                    imgs = X_train[batch_idx*batch_size:batch_idx*batch_size+batch_size, :]
                    labl = Y_train[batch_idx*batch_size:batch_idx*batch_size+batch_size, :]
                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    
                    #print (imgs.shape) #, idx
                    # Sample noise and generate a batch of new images
#                    noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
#                    #print(noise.shape, batch_size, self.latent_dim) #, noise[0,:]
#                    gen_imgs = self.generator.predict(noise)
#                    imgs_v(gen_imgs[0,:,:,:])
#                    # Train the discriminator (real classified as ones and generated as zeros)
#                    
#                    # ---------------------
#                    # обучаем дискриминатор на реальных изображениях
#                    # ---------------------
#                    d_loss_real = self.discriminator.train_on_batch(imgs, valid)
#                    
#                    # ---------------------
#                    # обучаем дискриминатор на реальных изображениях
#                    # ---------------------            
#                    d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
#                    
#                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

#                    # ---------------------
#                    #  Train Generator
#                    # ---------------------

#                    # Train the generator (wants discriminator to mistake images as real)
#                    g_loss = self.combined.train_on_batch(noise, valid)
#                    #if g_loss > 0.7:
#                    # Plot the progress
#                    print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
#                    #imgs_v(gen_imgs[0,:,:,:])


                    #Classif batch
                    #gen_imgs
                    lsss = self.model_res.train_on_batch(imgs, labl)
                    print(lsss[0], 100*lsss[1])
#                    aaa = self.model_res.predict(gen_imgs)
#                    for k in range(aaa.shape[0]):

#                            A = np.reshape(aaa[k], [-1, 9, len(CHARS)])
#                            best = np.argmax(A, 2)[0]
#                            answ = "".join(CHARS[int(i)] for i in best)
#                            genls = [sigmoid(A[:,il,L])[0] for il, L in enumerate(best)]
#                            print (answ, best, A.shape, sum(genls)/len(genls))
#                            if epoch > 50:
#                               if sum(genls)/len(genls) > 0.73:
#                                    cv2.imwrite("test_gen/"+str(k)+  "_"+str(answ)+".jpg", gen_imgs[k,:,:,:]*255.)#* 127.5 + 1.) 


                    # See class info
                    if batch_idx % 50 == 0:
                        idx = np.random.randint(0, X_test.shape[0], batch_size)
                        aaa = self.model_res.predict(X_test[idx])
                        #aaa = self.model_res.predict(X_test[:30])
                        for k in range(aaa.shape[0]):
                            correct = np.argmax(np.reshape(Y_test[idx][k], [-1, 9, len(CHARS)]), 2)
                            correct = "".join(CHARS[int(i)] for i in correct[0])
                            best = np.argmax(np.reshape(aaa[k], [-1, 9, len(CHARS)]), 2)
                            best = "".join(CHARS[int(i)] for i in best[0])
                            print (correct, "<->", best, correct == best)
                        print ("------------------------------>")
        except KeyboardInterrupt:
            print("Model saved")
#            for u in range(len(self.generator.get_weights())):
#                
#                   print (self.generator.get_weights()[u].shape)
            np.save('generator', np.array(self.generator.get_weights()))
            np.save('class', np.array(self.model_res.get_weights()))
#            model_json = self.generator.to_json()
#            with open("model.json", "w") as json_file:
#                 json_file.write(model_json)
#            self.generator.save_weights("model.h5")

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        print ("NOISE", noise.shape)
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=40000, batch_size=30, save_interval=50)
    