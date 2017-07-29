  # -*- coding: utf-8 -*-
"""
project:Tensorflow实现进阶卷积神经网络
@Author:Gene
@Github:https://github.com/Gene20/DeepLearning
@Email:GeneWithyou@gamil.com
@Website:www.gene20/top
"""  
"""
数据：CIFAR-10经典数据集 50000训练集+10000测试集，10类32x32的彩色图像
目的：图像识别
网络结构：Input-->Conv+ReLu+Pool+lrn-->Conv+ReLu+lrn+Pool-->FC+ReLu-->FC+ReLu-->FC-->Output
"""
import cifar10,cifar10_input
import tensorflow as tf
import numpy as np
import time

"""1.定义参数和算法公式"""
training_nums=3000
batch_size=128
data_dir='xxx?oooxoxoxox'
cifar10.maybe_download_and_extract()
train_images,train_labels=cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)
test_images,test_labels=cifar10_input.inputs(eval_data=True,data_dir=data_dir,batch_size=batch_size)
images_ph=tf.placehodler(tf.float32,[batch_size,24,24,3])
labels_ph=tf.placehodler(tf.int32,[batch_size])

"""2.定义权重"""
def weight_variable(shape,stddev,w1):
	var=tf.Variable(tf.truncated_normal(shape=shape,stddev=stddev))
	if w1 is not None:#对权重做L2正则化
		weight_loss=tf.multiply(tf.nn.l2_loss(var),w1,name='weight_loss')
		tf.add_to_collection('losses',weight_loss)
	return var
"""3.定义偏置"""
def bias_variable(val,shape):
	return tf.Variabel(tf.constant(val,shape=shape))

"""4.定义卷积"""
def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

"""5.定义池化"""
def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

"""6.第一个卷积层和池化层"""
W_conv1=weight_variable([5,5,3,64],stddev=5e-2,w1=0.0)
b_conv1=bias_variable(0.0,[64])
h_conv1=tf.nn.relu(conv2d(images_ph,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)
norm1=tf.nn.lrn(h_pool1,4,bias=1.0,alpha=0.001/9.0,beta=0.75)

"""7.第二个卷积层和池化层"""
W_conv2=weight_variable([5,5,64,64],stddev=5e-2,w1=0.0)
b_conv2=bias_variable(0.0,[64])
h_conv2=tf.nn.relu(conv2d(norm1,W_conv2)+b_conv2)
norm2=tf.nn.lrn(h_conv2,4,bias=1.0,alpha=0.001/9.0,beta=0.75)
h_pool2=max_pool_2x2(norm2)

"""8.全连接层"""
reshape=tf.reshape(h_pool2,[batch_size,-1])
dim=tf.get_shape(reshape)[1].value
W_fc1=weight_variable([dim,384],stddev=0.04,w1=0.004)
b_fc1=bias_variable(0.1,[384])
h_fc1=tf.nn.relu(tf.matmul(reshape,W_fc1)+b_fc1)

"""9.全连接层"""
W_fc2=weight_variable([184,192],stddev=0.04,w1=0.004)
b_fc2=bias_variable(0.1,[192])
h_fc2=tf.nn.relu(tf.matmul(h_fc1,W_fc2)+b_fc2)

"""10.模型inference输出结果"""
W_rs=weight_variable([192,10],stddev=1/192.0,w1=0.0)
b_rs=bias_variable(0.0,[10])
logits=tf.add(tf.matmul(h_fc2,W_rs),b_rs)

"""11.计算CNN的损失"""
def loss(logits,labels):
	labels=tf.cast(labels,tf.int64)
	cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits=logits,labels=labels,name='cross_entropy_per_example')
	cross_entropy_mean=tf.reduce_mean(cross_entropy,name='cross_entropy_mean')
	tf.add_to_collection('losses',cross_entropy_mean)
	return tf.add_n(tf.get_collection('losses'),name='total_loss')#对losses集合中的所有损失得到总损失并返回

"""12.计算损失和优化器"""
loss=loss(logits,labels_ph)
train_step=tf.train.AdamOptimizer(1e-3).minimize(loss)
#输出 top k 的准确率，默认为top1
topKAc=tf.nn.in_top_k(logits,labels_ph,1)
sess=tf.InteractiveSession()
tf.global_variabels_initializer().run()
#启动多线程
tf.train.start_queue_runners()

"""13.开始训练"""
for i in range(training_nums):
	start_time=time.time()
	images_batch,labels_batch=sess.run([images_ph,labels_ph])
	_,loss_value=sess.run([train_step,loss],feed_dict={images_ph:images_batch,labels_ph:labels_batch})

	duration=time.time()-start_time
	if i%10==0:
		exa_per_sec=batch_size/duration
		sec_pre_bat=float(duration)
		format_str='step: %d, losses= %.2f (%.1f examples/second & %.3f seconds/bacth)'
		print(format_str%(i,loss_value,exa_per_sec,sec_pre_bat))

"""14.开始测试"""
import math
num_examples=10000
num_iter=int(math.ceil(num_examples/batch_size))
num_total=num_iter*batch_size
true_count=0
for i in range(num_iter):
	images_batch,labels_batch=sess.run([images_test,labels_test])
	predictions=sess.run([topKAc],feed_dict={images_ph:images_test,labels_ph:labels_batch})
	true_count+=math.sum(predictions)
print('Accuarcy @Top 1 is :%.3f'%(true_count/num_total))




