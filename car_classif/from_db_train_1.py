import os
import cv2
import numpy as np
import itertools
from mongodb import *
import tensorflow as tf
import random
import multiprocessing
import functools
import time
slim = tf.contrib.slim

def imgs(x):
      print (x.shape)
      cv2.imshow('Rotat', np.array(x))
      cv2.waitKey(0)
      #time.sleep(0.05)
      cv2.destroyAllWindows()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def res_net_block(input_data, filter_count):
  x = slim.conv2d(input_data, filter_count, (1, 1), padding="SAME", activation_fn=tf.nn.relu)
  x = slim.conv2d(x, filter_count, (3, 3), padding="SAME", activation_fn=None)
  x = tf.nn.relu(x)
  x = slim.conv2d(x, filter_count, (1, 1), padding="SAME", activation_fn=None)
  x = tf.add(x, input_data)
  x = tf.nn.relu(x)
  return x


def convolutional_layers():
    """
    Get the convolutional layers of the model.
    """
    x_ = tf.placeholder(tf.float32, [None, 240, 320, 3])
    x_expanded = x_#tf.expand_dims(x_, 3)
    is_training = tf.placeholder(tf.bool, [])
    print ("IN", x_expanded.shape)
    # First layer
    strides = 1
    with slim.arg_scope([slim.conv2d],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training}):
        x = slim.conv2d(x_expanded, 32, (3, 3), stride=strides, padding="SAME", activation_fn=tf.nn.relu) 
        x = slim.max_pool2d(x, kernel_size=2, stride=2)
        print (x.shape)
        x = slim.conv2d(x, 64, (3, 3), stride=strides, padding="SAME", activation_fn=tf.nn.relu) 
        x = slim.max_pool2d(x, kernel_size=2, stride=2)        
        x = res_net_block(x, 64)
        print (x.shape)
        x = slim.conv2d(x, 64, (3, 3), stride=strides, padding="SAME", activation_fn=tf.nn.relu) 
        x = slim.max_pool2d(x, kernel_size=2, stride=2)
        x = res_net_block(x, 64)    
        print (x.shape)
        x = slim.conv2d(x, 64, (3, 3), stride=strides, padding="SAME", activation_fn=tf.nn.relu) 
        x = slim.conv2d(x, 64, (3, 3), stride=strides, padding="VALID", activation_fn=tf.nn.relu) 
        x = res_net_block(x, 64)
        x = slim.conv2d(x, 64, (2, 2), stride=strides, padding="VALID", activation_fn=tf.nn.relu) 
        x = slim.conv2d(x, 64, (3, 3), stride=strides, padding="VALID", activation_fn=tf.nn.relu) 
        print (x.shape)  
        x = res_net_block(x, 64)
        x = slim.conv2d(x, 64, (2, 2), stride=strides, padding="VALID", activation_fn=tf.nn.relu)  
        x = slim.max_pool2d(x, kernel_size=2, stride=2)        
        x = res_net_block(x, 64)
        x = slim.conv2d(x, 64, (2, 2), stride=strides, padding="VALID", activation_fn=tf.nn.relu) 
        x = slim.max_pool2d(x, kernel_size=2, stride=2)  
        print (x.shape)
        # Densely connected layer
        W_fc1 = weight_variable([5 * 8 * 64, 1024])
        b_fc1 = bias_variable([1024])
        conv_layer_flat = tf.reshape(x, [-1, 5 * 8 * 64])
        print (conv_layer_flat.shape)
        h_fc1 = tf.nn.relu(tf.matmul(conv_layer_flat, W_fc1) + b_fc1)
     
        W_fc2 = weight_variable([1024, 35])
        b_fc2 = bias_variable([35])

        y = tf.matmul(h_fc1, W_fc2) + b_fc2
        print (y.shape)
        
        return x_, y, is_training


#def get_loss(y, y_):
#    # Calculate the loss from digits being incorrect.  Don't count loss from
#    # digits that are in non-present plates.
#    digits_loss = tf.nn.softmax_cross_entropy_with_logits(
#                                          logits=tf.reshape(y[:, 1:],
#                                                     [-1, len(common.CHARS)]),
#                                          labels=tf.reshape(y_[:, 1:],
#                                                     [-1, len(common.CHARS)]))
#    digits_loss = tf.reshape(digits_loss, [-1, 9])
#    digits_loss = tf.reduce_sum(digits_loss, 1)
#    digits_loss *= (y_[:, 0] != 0)
#    digits_loss = tf.reduce_sum(digits_loss)

#    # Calculate the loss from presence indicator being wrong.
#    presence_loss = tf.nn.sigmoid_cross_entropy_with_logits(
#                                                          logits=y[:, :1], labels=y_[:, :1])
#    presence_loss = 9 * tf.reduce_sum(presence_loss)

#    return digits_loss, presence_loss, digits_loss + presence_loss





#data = DataBase("anpr")
db.create_collection("anpr")
list_temp = db.see_all_post()
list_T = list_temp[:100]
list_temp = list_temp[100:] 
random.shuffle(list_temp)

def code_to_vec(c, a):
        y = np.zeros((len(a),))
        y[a.index(c)] = 1.0
        return y.flatten()

def create_vec():
    temp_dict_mark = {}
    vector_model = []
    for i in list_temp:
        if i['model'] not in vector_model:
            vector_model.append(i['model']) 
        try:
            temp_dict_mark[i['mark']].append(i['model'])
  #########?????????????????????
        except:
            temp_dict_mark[i['mark']] = [i['model']]
    return temp_dict_mark, vector_model       


#print (list_temp)
def imgs_data_read():
        #db = DataBase("anpr")
        db.create_collection("anpr") 
        for i in db.see_all_post():
               print (i)
            #if i["name"] == "name":
               img = db.file.get(i["images"]).read()
               nparr = np.fromstring(img, np.uint8)
               img_t = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
               imgs(img_t)

def unzip(b):
    xs, ys = zip(*b)
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys

def mpgen(f):
    def main(q, args, kwargs):
        try:
            for item in f(*args, **kwargs):
                q.put(item)
        finally:
            q.close()

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        q = multiprocessing.Queue(10) 
        proc = multiprocessing.Process(target=main,
                                       args=(q, args, kwargs))
        proc.start()
        try:
            while True:
                item = q.get()
                yield item
        finally:
            proc.terminate()
            proc.join()

    return wrapped
        

@mpgen
def read_batches(batch_size):
      #pass
      def gen_vecs():
          L = itertools.islice(list_temp, batch_size)
          for im in L:
                 img = db.file.get(im["images"]).read()
                 nparr = np.fromstring(img, np.uint8)
                 img_t = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                 img_t = cv2.resize(img_t,(320, 240))#240, 320
                 #print (im["mark"], im["model"],  code_to_vec(im["mark"], vector_mark), code_to_vec(im["model"], vector_model[1]).shape)
                 #yield "a","a"
                 cs = code_to_vec(im["mark"], vector_mark)
                 cs2 = code_to_vec(im["model"], vector_model[1])
                 #print (cs.shape, cs2.shape)
                 #cs_a = np.concatenate([cs, cs2])
                 cs_a = cs
                 #imgs(img_t)
                 #print (im["model"], im["mark"], img_t.shape, cs_a.shape, cs.shape, cs2.shape)
                 yield img_t, cs_a#"a", "b"#im.astype(numpy.float32) / 255., code_to_vec(p, c)
      #yield im.astype(numpy.float32) / 255., code_to_vec(p, c)
      while True:
             yield unzip(gen_vecs())
#def numpy_vec(): 
 #-------------------------------------------------->
vector_model = create_vec()
vector_mark = list(vector_model[0].keys())
print (vector_mark)
#cs = code_to_vec("audi", vector_mark)
#cs2 = code_to_vec("astra", vector_model[1])
#cs_a = np.concatenate([cs, cs2])

#print (vector_mark, cs_a.shape, cs.shape[0], cs2.shape[0])#ls[1] cs2, cs  
#-------------------------------------------------->  
#while True:
#batch_iter = enumerate(read_batches(40))
##print (len(list(batch_iter)))
#for batch_idx, (batch_xs, batch_ys) in batch_iter:
#                print ("---->", batch_idx, batch_xs.shape, batch_ys.shape)
                #print (batch_xs[0].shape, batch_ys[0].shape)
       
# 640 480  
#-------------------------------------------------->
#data = DataBase("anpr")
#print (dir(data))
#print (data.see_collection())
#db.create_collection("anpr") 
#list_temp = db.see_all_post()
#myls = ['audi']
##print (len())
#temp_dict_mark = {}
#for i  in list_temp:
#    try:
#      if i['model'] not in temp_dict_mark[i['mark']]:
#        temp_dict_mark[i['mark']].append(i['model'])
#    except:
#        temp_dict_mark[i['mark']] = [i['model']]
#print (len(temp_dict_mark.keys()))
###print (temp_dict_mark.keys(), temp_dict_mark['audi'])
#temp_list = []
#dict_to_t = {}
#for i in temp_dict_mark:
#    print (i, len(temp_dict_mark[i])) #temp_dict_mark[i],
#    p = 0
#    dict_to_t[i] = {"len_models":len(temp_dict_mark[i]), "list_len_models":[]}
#    for x in range(len(temp_dict_mark[i])):
#        p += 1
#        dict_to_t[i]["list_len_models"].append(len(db.find_many_post({"model":temp_dict_mark[i][x]})))# db.count1
#        #temp_list.append(len(db.find_many_post({"model":temp_dict_mark[i][x]})))
#        #print ()#()
##print (max(temp_list), min(temp_list))
#all_len = 0
#for P in dict_to_t:
#    all_len += dict_to_t[P]["len_models"]#len(dict_to_t[P]["list_len_models"])
#    print (dict_to_t[P], len(dict_to_t[P]["list_len_models"]))

#print (all_len)
##---------------------------------->


# 35 - 30

#1029 + 35
# 1064
# vector
 
   
#Взять все модели и марки
#Отсортировать модели и марки
#Обучать 2 вектора 
# 1...N , 1...N    
# Узнать самую большую по кол ву
# моделей

#def get_loss(y, y_):
#    # Calculate the loss from digits being incorrect.  Don't count loss from
#    # digits that are in non-present plates.
#    #1029 - 35
#    digits_loss = tf.nn.softmax_cross_entropy_with_logits(logits=tf.reshape(y[:, 1010:], [-1, 35]),
#                                                          labels=tf.reshape(y_[:, 1010:], [-1, 35]))
#    digits_loss = tf.reshape(digits_loss, [-1, 35])
#    digits_loss = tf.reduce_sum(digits_loss, 1)
#    digits_loss = tf.reduce_sum(digits_loss)

#    # Calculate the loss from presence indicator being wrong.
##    presence_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y[:, :1], labels=y_[:, :1])
##    presence_loss = 9 * tf.reduce_sum(presence_loss)
#    digits_loss1 = tf.nn.softmax_cross_entropy_with_logits(logits=tf.reshape(y[:, :1010], [-1, 1010]),
#                                                           labels=tf.reshape(y_[:, :1010], [-1,1010]))
#    digits_loss1 = tf.reshape(digits_loss1, [-1, 1010])
#    digits_loss1 = tf.reduce_sum(digits_loss1, 1)
#    digits_loss1 = tf.reduce_sum(digits_loss1)


#    return digits_loss, digits_loss1, digits_loss + digits_loss1

def get_loss(y, y_):
    # Calculate the loss from digits being incorrect.  Don't count loss from
    # digits that are in non-present plates.
    #1029 - 35
#    digits_loss = tf.nn.softmax_cross_entropy_with_logits(logits=y[:, :],
#                                                          labels=y_[:, :])
#    digits_loss = tf.reshape(digits_loss, [-1, 35])
#    digits_loss = tf.reduce_sum(digits_loss, 1)
#    digits_loss = tf.reduce_sum(digits_loss)
    digits_loss = tf.nn.softmax_cross_entropy_with_logits(
                                          logits=tf.reshape(y[:, :],
                                                     [-1, 35]),
                                          labels=tf.reshape(y_[:, :],
                                                     [-1, 35]))
    digits_loss = tf.reshape(digits_loss, [-1, 1])
    digits_loss = tf.reduce_sum(digits_loss, 1)


    return digits_loss


def train(learn_rate, report_steps, batch_size, initial_weights=None):        
    x, y, is_training = convolutional_layers()
    print (x, y, is_training)   
    y_ = tf.placeholder(tf.float32, [None, 35])
    #digits_loss, digits_loss1, loss = get_loss(y, y_)
    loss = get_loss(y, y_)
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)
    
    best = tf.argmax(tf.reshape(y[:, :], [-1, 1, 35]), 2)
    correct = tf.argmax(tf.reshape(y_[:, :], [-1, 1, 35]), 2)
#    
#    best1 = tf.argmax(tf.reshape(y[:, 1010:], [-1, 1, 35]), 2)
#    correct1 = tf.argmax(tf.reshape(y_[:, 1010:], [-1, 1, 35]), 2)    

    init = tf.initialize_all_variables()    
    def get_test():
        L = itertools.islice(list_T, batch_size)
        LS0 = []
        LS1 = []
        for im in L:
                     img = db.file.get(im["images"]).read()
                     nparr = np.fromstring(img, np.uint8)
                     img_t = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                     img_t = cv2.resize(img_t,(320, 240))#240, 320
                     cs = code_to_vec(im["mark"], vector_mark)
                     cs_a = cs
                     #imgs(img_t)
                     LS0.append(img_t)
                     LS1.append(cs_a)
        return np.array(LS0), np.array(LS1)
     
    test_xs, test_ys = get_test()
    #print (test_xs.shape)
    #imgs(test_xs[10]) 
    def do_report():
        r = sess.run([best, correct, loss], feed_dict={x: test_xs, y_: test_ys, is_training:"False"})
        print (vector_mark[int(r[1][0])], vector_mark[int(r[0][0])], r[2][0])#r[0].shape,r[1].shape
        #imgs(test_xs[0])
    
    def do_batch():
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, is_training:"True"}) 
#        if batch_idx % report_steps == 0:
#            do_report()
        
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.60)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if initial_weights is not None:
            saver.restore(sess, "model_slim/model.ckpt")
        else:    
            sess.run(init)
         
        print ("START")
        try:
            last_batch_idx = 0
            last_batch_time = time.time()
            batch_iter = enumerate(read_batches(batch_size))
            for batch_idx, (batch_xs, batch_ys) in batch_iter:
                do_batch()
                if batch_idx % report_steps == 0:
                     print (batch_idx, batch_xs.shape, batch_ys.shape)#, batch_ys[0].shape
                     do_report()
                     #------------>
#                    batch_time = time.time()
#                    if last_batch_idx != batch_idx:
#                        print ("time for "+str(batch_size)+" batches {}".format(
#                            batch_size * (last_batch_time - batch_time) /
#                                            (last_batch_idx - batch_idx)))
#                        last_batch_idx = batch_idx
#                        last_batch_time = batch_time

        except KeyboardInterrupt:
             save_path = saver.save(sess, "model_slim/model.ckpt")
             print("Model saved in path")# % save_path)
             
train(learn_rate=0.001,
          report_steps=100,
          batch_size=40,
          initial_weights=1) #initial_weights=None) 

