#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: JX
# data: 20181101

import tensorflow as tf
import numpy as np
import os
import glob
import cv2
import sys

image_size = 64
num_channels = 3
images = []

path = 'cat.1.jpg'
image = cv2.imread(path)
image = cv2.resize(image,(image_size,image_size),0,0,cv2.INTER_LINEAR)
images.append(image)
images = np.array(images,dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images,1.0/255.0)
x_batch = images.reshape(1,image_size,image_size,num_channels)

sess = tf.Session()
saver = tf.train.import_meta_graph('./dogs-cats-model/dog-cat.ckpt-9975.meta')
saver.restore(sess,'./dogs-cats-model/dog-cat.ckpt-9975')
graph = tf.get_default_graph()

y_pred = graph.get_tensor_by_name("y_pred:0")

x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((1,2))

feed_dict_testing = {x:x_batch,y_true:y_test_images}
result = sess.run(y_pred,feed_dict=feed_dict_testing)

res_label = ['dog','cat']
print(res_label[result.argmax()])