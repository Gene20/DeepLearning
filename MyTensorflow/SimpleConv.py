# -*- coding: utf-8 -*-
"""
project:5.2Tensorflow实现简单卷积网络
@Author:Gene
@Github:https://github.com/Gene20/DeepLearning
@Email:GeneWithyou@gamil.com
@Website:www.gene20/top
"""
"""
卷积神经网络：两个卷积层+一个全连接层
目的：手写数字识别
数据集：MNIST数据
预计准确率：99.2%
结构：input-->Conv1+ReLu-->MaxPool1-->Conv2+ReLu-->MaxPool2-->FC1+ReLu-->DropOut-->FC2+SoftMax-->Output
"""
from tensroflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)
sess=tf.InteractiveSession()

"""定义权重和偏置"""
def weight_variable(shape):
	return tf.Variable(tf.truncated_normal(shape=shape,stddev=0.1))#截断的高斯分布
def bias_variable(shape):
	return tf.Variabel(tf.constant(0.1,shape=shape))
"""定义卷积"""
def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
"""定义池化"""
def max_pool_2x2(x):
	return tf.nn.max_pool_2x2(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
"""定义参数和算法公式"""
input_n=784
training_nums=20000
batch_size=50
learning_rate=1e-4
x=tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)
x_images=tf.reshape(x,[-1,28,28,1])

"""第一个卷积层"""
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_images,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)

"""第二个卷积层"""
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)

"""全连接层"""
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2,W_fc1)+b_fc1)

"""DropOut层"""
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

"""全连接层"""
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
y=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

"""定义损失和优化器"""
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
train_step=tf.AdamOptimizer(learning_rate).minimize(cross_entropy)
predictions=tf.eqaul(tf.argmax(y_,1),tf.argmax(y,1))
accuarcy=tf.reduce_mean(tf.cast(predictions,tf.float32))

"""训练过程"""
tf.global_variables_initializer().run()
for i in range(training_nums):
	x_batch,y_batch=tf.train.next_batch(batch_size)
	if i%100==0:
		print('step: %d, accuarcy: %g'%(i,accuarcy.eval(feed_dict={x:x_batch,y_:y_batch,keep_prob:1.0})))
	train_step.run(feed_dict={x:x_batch,y_:y_batch,keep_prob:0.5})
"""测试过程"""
print('accuarcy: %g'%accuarcy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))
"""
后记：这个CNN模型可以得到准确率约为99.2%，基本满足对手写数字识别准确率的要求了
"""  	



