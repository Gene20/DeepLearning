# -*- coding: utf-8 -*-
"""
project:TensorFlow实现多层感知机
@Author:Gene
@Github:https://github.com/Gene20/DeepLearning
@Email:GeneWithyou@gamil.com
@Website:www.gene20/top
"""
"""
全连接神经网络结构：
Input-->(w*x+b)-->ReLu激活函数-->DropOut层-->(w*x+b)-->softmax层-->Output
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)
sess=tf.InteractiveSession()

"""神经网络第一步：定义算法公式(神经网络Forward)"""
in_units=784
h1_units=300
w1=tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))#截断的高斯分布/正态分布
b1=tf.Variable(tf.zeros([h1_units]))
w2=tf.Variable(tf.zeros([h1_units,10]))
b2=tf.Variable(tf.zeros([10]))

x=tf.placeholder(tf.float32,[None,in_units])
keep_prob=tf.placeholder(tf.float32)

hidden1=tf.nn.relu(tf.matmul(x,w1)+b1)
hidden1_drop=tf.nn.dropout(hidden1,keep_prob)
y=tf.nn.softmax(tf.matmul(hidden1_drop,w2)+b2)

"""神经网络第二步：定义损失函数和选择优化器来优化loss（损失)"""
y_=tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
train_step=tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

"""神经网络第三步：训练步骤"""
tf.global_variables_initializer().run()
for i in range(3000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    train_step.run(feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.75})

"""4.神经网络第四步：对模型进行准确率评测"""
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuarcy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(accuarcy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))



    



   