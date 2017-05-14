# -*- coding: utf-8 -*-
"""
Project:
@Author:Gene
@Email:GeneWithyou@gmail.com
@Github:github.com/Gene20/DeepLearning
@PersonalWeb:www.gene20.top
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile
import tensorflow as tf
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import tensorflow.examples.tutorials.mnist
mnist = read_data_sets("MNIST_data/", one_hot=True)

#预测
x=tf.placeholder(tf.float32,[None,784])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,W)+b)
#交叉熵
y_=tf.placeholder('float',[None,10])
cross_entropy=-tf.reduce_sum(y_*tf.log(y))
#梯度下降算法
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#初始化
init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)
#训练模型
for i in range(1000):
    x_batch,y_batch=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:x_batch,y_:y_batch})
#预测正确率
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))
#打印正确率
print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
