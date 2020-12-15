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
import tensorflow as tf

import matplotlib.pyplot as plt

import sys, cv2, os
import json

import numpy as np
from gen_rus_plate4 import GenPlate

#from gen_all_1 import *
import gen_all_1

KL = GenPlate()
CHARS = KL.LETTER + KL.DIGIT + " "


def imgs_v(x):
      cv2.imshow('Rotat', np.array(x))#, dtype=np.uint8
      cv2.waitKey(1)
      #cv2.destroyAllWindows()

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

        model.add(Dense(9*len(CHARS), activation='softmax'))
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
            imgh = cv2.imread(gen_all_1.H4.file[j][0])
            imgh = cv2.resize(imgh, (128, 64))
            #imgs_v(imgh[:,:,0])
            #print (imgh.shape)
            json_open = json.load(open(gen_all_1.H4.label[j][0], "r"))
            vec_arr = code_to_vec(json_open["description"])
            #print (vec_arr, np.reshape(vec_arr,(9,22)))
            arr_rus_lab.append(vec_arr)
            arr_rus.append(imgh[:,:,:1])
        return np.array(arr_rus), np.array(arr_rus_lab)


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

    def train(self, epochs, batch_size=30, save_interval=50):
        plate_rus = gen_all_1.H4.label.keys()
        x_train, y_train = get_data(plate_rus)
        #################

        # Train
        X_train, Y_train = x_train[100:], y_train[100:]
        # TEST
        X_test, Y_test = x_train[:100], y_train[:100]
        #print (Y_train.shape, np.random.randint(0, len(plate_rus), batch_size), Y_train[0])
        #self.model_res
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        try:
            for epoch in range(epochs):
                for batch_idx in range(X_train.shape[0]//batch_size):
                     
                    #idx = np.random.randint(0, len(plate_rus), batch_size)
                    #imgs = X_train[idx]
                    #labl = Y_train[idx]
                    #print (len(idx), X_train[idx].shape, X_train[batch_idx*batch_size:batch_idx*batch_size+batch_size, :].shape)
                    imgs = X_train[batch_idx*batch_size:batch_idx*batch_size+batch_size, :]
                    labl = Y_train[batch_idx*batch_size:batch_idx*batch_size+batch_size, :]
                    lsss = self.model_res.train_on_batch(imgs, labl)
                    #print (lsss)

                    # ---------------------
                    #  Train Discriminator
                    # ---------------------

                    # Select a random half of images
                    #idx = np.random.randint(0, len(plate_rus), batch_size)
                    #imgs = X_train[idx]
                    print (imgs.shape) #, idx
                    # Sample noise and generate a batch of new images
                    noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                    #print(noise.shape, batch_size, self.latent_dim) #, noise[0,:]
                    gen_imgs = self.generator.predict(noise)
                    #imgs_v(imgs[0,:,:,:])
                    # Train the discriminator (real classified as ones and generated as zeros)
                    
                    # ---------------------
                    # обучаем дискриминатор на реальных изображениях
                    # ---------------------
                    d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                    
                    # ---------------------
                    # обучаем дискриминатор на реальных изображениях
                    # ---------------------            
                    d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                    
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                    # ---------------------
                    #  Train Generator
                    # ---------------------

                    # Train the generator (wants discriminator to mistake images as real)
                    g_loss = self.combined.train_on_batch(noise, valid)
                    #if g_loss > 0.7:
                    # Plot the progress
                    print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

                    # ---------------------
                    #  ???
                    # ---------------------
                    imgs_v(gen_imgs[0,:,:,:])

                    if batch_idx % 50 == 0:
    #                    idx = np.random.randint(0, 100, batch_size)
    #                    aaa = self.model_res.predict(X_test[idx])
                        aaa = self.model_res.predict(X_test[:30])
                        #for k in range(len(idx)):
                        for k in range(aaa.shape[0]):
                            correct = np.argmax(np.reshape(Y_test[:30][k], [-1, 9, len(CHARS)]), 2)
                            best = np.argmax(np.reshape(aaa[k], [-1, 9, len(CHARS)]), 2)

                            print ("".join(CHARS[int(i)] for i in correct[0]), "<->", "".join(CHARS[int(i)] for i in best[0]))
                        # serialize model to JSON
                        print ("------------------------------>")
        except KeyboardInterrupt:
            print("Model saved")
                    #,X_train.shape[0]//batch_size, X_train.shape[0], batch_size, batch_idx, epoch)
            model_json = self.model_res.to_json()
            with open("model.json", "w") as json_file:
                 json_file.write(model_json)
            self.model_res.save_weights("model.h5")
                    #print (Y_train[idx][0].shape, s_ar.shape)#"".join(CHARS[int(i)] for i in Y_train[idx][0])
            #print (Y_train.shape)
            #history = self.model_res.fit(X_train, Y_train, epochs=100, verbose=0)
                #imgs_v(imgs)
                #print (epoch, imgs.shape)

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
    #print(len(gen_all_1.H4.label.keys()))
    
    print (CHARS)

#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
#-------------------->
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk")
#Подавать изображения цифр и ожидать появление реального изображения
# Кластеризация цифр на изображении 
    
# Подключить классифиакатор    
