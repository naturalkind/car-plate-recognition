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


import matplotlib.pyplot as plt

import sys, cv2, os
import json

import numpy as np
from gen_rus_plate4 import GenPlate

#from gen_all_1 import *
import gen_all_1

KL = GenPlate()
CHARS = KL.LETTER + KL.DIGIT


def imgs_v(x):
      cv2.imshow('Rotat', np.array(x))#, dtype=np.uint8
      cv2.waitKey(1)
      #cv2.destroyAllWindows()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
def res_net_block(input_data, filter_count):
  x = layers.Conv2D(filter_count, (1, 1), activation='relu', padding='same')(input_data)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2D(filter_count, (3, 3), activation=None, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = tf.nn.relu(x)
  x = layers.Conv2D(filter_count, (1, 1), activation=None, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Add()([x, input_data])
  x = tf.nn.relu(x)#x = layers.Activation('relu')(x)
  print (x.shape)
  return x


def convolutional_layers():
    """
    Get the convolutional layers of the model.
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='max1'))
    model.add(Conv2D(64, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='max2'))
    model.add(res_net_block(x, 64))
    model.add(Conv2D(128, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='max3'))
    model.add(res_net_block(x, 128))
    model.add(Conv2D(128, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='valid', name='conv5', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(res_net_block(x, 256))
    model.add(Conv2D(256, (2, 2), padding='valid', name='conv6', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='valid', name='conv7', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(res_net_block(x, 256))
    model.add(Conv2D(256, (2, 2), padding='valid', name='conv8', kernel_initializer='he_normal'))
#    model.add()
#    model.add()
#    model.add()
#    model.add()
#    model.add()
#    model.add()
#    model.add()
#    model.add()
               
#    x = MaxPooling2D(pool_size=(1, 2), name='max4')(x)
#    print (x.shape)
    
    
      
    x = BatchNormalization()(x)      
    print (x.shape)
    # Densely connected layer
    W_fc1 = weight_variable([2 * 10 * 256, 2048])
    b_fc1 = bias_variable([2048])
    conv_layer_flat = tf.reshape(x, [-1, 2 * 10 * 256])
    h_fc1 = tf.nn.relu(tf.matmul(conv_layer_flat, W_fc1) + b_fc1)
    model.add(h_fc1)
 
    W_fc2 = weight_variable([2048, 1 + 9 * len(CHARS)])
    b_fc2 = bias_variable([1 + 9 * len(CHARS)])

    y = tf.matmul(h_fc1, W_fc2) + b_fc2
    model.add(y)
    model.summary()
    img = Input(shape=(64,128,1))
    validity = model(img)
    return Model(img, validity)   

#Сделать шум от 0 до 36 
#для вектора 1x100 

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 64
        self.img_cols = 128
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 192#40#36#100

        optimizer = Adam(0.0002, 0.5)

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
        
        #----------------->
        #model_res = convolutional_layers()
        
        

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

    def train(self, epochs, batch_size=30, save_interval=50):#128

        # Load the dataset
        plate_rus = gen_all_1.H4.label.keys()
        arr_rus = []
        arr_rus_lab = []
        def code_to_vec(code):
            def char_to_vec(c):
                y = np.zeros((len(CHARS),))
                #print c
                y[CHARS.index(c)] = 1.0
                return y

            c = np.vstack([char_to_vec(c) for c in code])
            #print (c.shape)
            return c.flatten()
        for j in plate_rus:
            imgh = cv2.imread(gen_all_1.H4.file[j][0])
            imgh = cv2.resize(imgh, (128, 64))
            #imgs_v(imgh[:,:,0])
            #print (imgh.shape)
            json_open = json.load(open(gen_all_1.H4.label[j][0], "r"))
            vec_arr = code_to_vec(json_open["description"])
            #print (vec_arr, np.reshape(vec_arr,(9,22)))
            arr_rus_lab.append(vec_arr)
            arr_rus.append(imgh[:,:,0])
        #plate_rus = gen_all_1.generate_ims()
        #---
        (X_train, _YY), (_, _) = mnist.load_data()
        #print (X_train.shape)
        t_arr = np.array(arr_rus)
        arr_rus_lab = np.array(arr_rus_lab)
        X_train = t_arr
        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        #print (len(plate_rus), X_train.shape, t_arr.shape, arr_rus_lab.shape, arr_rus_lab[0])
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, len(plate_rus), batch_size)
            imgs = X_train[idx]
            print (imgs.shape, _YY[idx][0]) #, idx
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
            #
            # обучить классификатор на реальных данных
           
            # If at save interval => save generated image samples
            #if epoch % save_interval == 0:
            #    self.save_imgs(epoch)

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

#Подавать изображения цифр и ожидать появление реального изображения
# Кластеризация цифр на изображении 
    
# Подключить классифиакатор    
