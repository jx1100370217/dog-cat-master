#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: JX
# data: 20190118
import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

cwd = "D://Code//pycharm//learning//20181101//training_data//"
classes = {'cats','dogs'}
writer = tf.python_io.TFRecordWriter('train.tfrecords')

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value= [value]))

for index,name in enumerate(classes):
    class_path = cwd + name + '//'
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name

        img = Image.open(img_path)
        img = img.resize((208,208))
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": _int64_feature(index),
            "img_raw": _bytes_feature(img_raw),
        }))
        writer.write(example.SerializeToString())  # 序列化为字符串
writer.close()

def read_and_decode(filename, batch_size): # read train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])# create a queue

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#return file_name and file
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })#return image and label

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [208, 208, 3])  #reshape image to 512*80*3
#    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 #throw img tensor
    label = tf.cast(features['label'], tf.int32) #throw label tensor

    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size= batch_size,
                                                    num_threads=64,
                                                    capacity=2000,
                                                    min_after_dequeue=1500,
                                                    )
    return img_batch, tf.reshape(label_batch,[batch_size])

tfrecords_file = 'D://Code//pycharm//learning//20181101//train.tfrecords'
BATCH_SIZE = 4
image_batch, label_batch = read_and_decode(tfrecords_file, BATCH_SIZE)

with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop() and i < 1:
            # just plot one batch size
            image, label = sess.run([image_batch, label_batch])
            for j in np.arange(BATCH_SIZE):
                print('label: %d' % label[j])
                plt.imshow(image[j, :, :, :])
                plt.show()
            i += 1
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)






























