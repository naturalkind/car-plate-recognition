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
    x_ = tf.placeholder(tf.float32, [None, 64, 128])
    x_expanded = tf.expand_dims(x_, 3)
    print ("IN", x_expanded.shape)
    # First layer
    x = Conv2D(32, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(x_expanded)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='max1')(x)
    print (x.shape)
    x = Conv2D(64, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='max2')(x)
    print (x.shape)
    x = res_net_block(x, 64)
    print (x.shape)
    x = Conv2D(128, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='max3')(x)
    print (x.shape)
    x = res_net_block(x, 128)    
    print (x.shape)
    x = Conv2D(128, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
#    x = MaxPooling2D(pool_size=(1, 2), name='max4')(x)
#    print (x.shape)
    x = Conv2D(256, (3, 3), padding='valid', name='conv5', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)  
    
    x = res_net_block(x, 256)
    
    x = Conv2D(256, (2, 2), padding='valid', name='conv6', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)   
    
    x = Conv2D(256, (3, 3), padding='valid', name='conv7', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)           
      
    x = res_net_block(x, 256)
    x = Conv2D(256, (2, 2), padding='valid', name='conv8', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)      
    print (x.shape)
    # Densely connected layer
    W_fc1 = weight_variable([2 * 10 * 256, 2048])
    b_fc1 = bias_variable([2048])
    conv_layer_flat = tf.reshape(x, [-1, 2 * 10 * 256])
    h_fc1 = tf.nn.relu(tf.matmul(conv_layer_flat, W_fc1) + b_fc1)
 
    W_fc2 = weight_variable([2048, 1 + 9 * len(CHARS)])
    b_fc2 = bias_variable([1 + 9 * len(CHARS)])

    y = tf.matmul(h_fc1, W_fc2) + b_fc2
    return x_, y

#
def build_discriminator():

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
        self.model_res = build_discriminator()
        self.model_res.compile(loss='categorical_crossentropy',
                               optimizer=optimizer,
                               metrics=['accuracy'])
        #self.model_res = convolutional_layers()

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
