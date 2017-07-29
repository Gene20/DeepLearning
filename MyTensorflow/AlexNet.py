  # -*- coding: utf-8 -*-
"""
project:Tensorflow实现ALexNet卷积神经
@Author:Gene
@Github:https://github.com/Gene20/DeepLearning
@Email:GeneWithyou@gamil.com
@Website:www.gene20/top
"""  
"""
网络结构：Input-->Conv+ReLu+LRN+Pool-->Conv+ReL+LRN+Pool-->Conv+ReLu-->Conv+ReLu-->Conv+ReLu-->Conv+ReLu
-->Pool-->FC+ReLu-->FC+ReLu-->FC+Softmax-->Output
"""
from datetime import datetime
import time
import math
import tensorflow as tf
batch_size=32
batch_nums=100
"""1.显示神经网络每一层结构"""
def print_activation(t):
	print(t.op.name,'',t.get_shape().as_list())

""""2.定义权重和偏置"""
def weight_variables(shape,dtype=tf.float32,stddev,name):
	return tf.Variable(tf.truncated_normal(shape=shape,dtype=dtype,stddev=stddev),name=name)
def bias_variables(val,shape,dtype=tf.float32,tb=True,name):
	return tf.Variable(tf.constant(val,shape=shape,dtype=dtype),trainable=tb,name=name)
"""3.定义卷积和池化"""
def conv2d(x,W,strides,padding='SAME'):
	return tf.conv2d(x,W,strides=strides,padding=padding)
def max_pool_2x2(x,ksize,strides,padding,name):
	return tf.max_pool(x,ksize=ksize,strides=strides,padding=padding,name=name)
"""4.定义inference函数"""
def inference(images):
	parameters=[]
	"""第一个卷积层和池化层"""
	with tf.name_scope('conv1') as scope:
		W_conv1=weight_variables([11,11,3,64],tf.float32,1e-1,'weights')
		b_conv1=bias_variables(0.0,[64],tf.float32,True,'bias')
		h_conv1=tf.nn.relu(tf.nn.bias_add(conv2d(images,W_conv1,[1,4,4,1],'SAME'),b_conv1),name=scope)
		print_activation(h_conv1)
		parameters+=[W_conv1,b_conv1]
	"""LRN层和池化层"""
	lrn1=tf.nn.lrn(h_conv1,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn1')
	h_pool1=max_pool_2x2(lrn1,[1,3,3,1],[1,2,2,1],'VALID','pool1')
	print_activation(h_pool1)
	"""第二个卷积层和池化层"""
	with tf.name_scope('conv2')as scope:
		W_conv2=weight_variables([5,5,64,192],tf.float32,1e-1,'weights')
		b_conv2=bias_variables(0.0,[192],tf.float32,True,'bias')
		h_conv2=tf.nn.relu(tf.nn.bias_add(conv2d(h_pool1,W_conv2,[1,1,1,1],'SAME'),b_conv2),name=scope)
		print_activation(h_conv2)
		parameters+=[W_conv2,b_conv2]
	"""LRN层和池化层"""
	lrn2=tf.nn.lrn(h_conv2,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn2')
	h_pool2=max_pool_2x2(lrn2,[1,3,3,1],[1,2,2,1],'VALID','pool2')
	print_activation(h_pool2)
	"""第三个卷积层和池化层"""
	with tf.name_scope('conv3')as scope:
		W_conv3=weight_variables([3,3,192,384],tf.float32,1e-1,'weights')
		b_conv3=bias_variables(0.0,[384],tf.float32,True,'bias')
		h_conv3=tf.nn.relu(tf.nn.bias_add(conv2d(h_pool2,W_conv3,[1,1,1,1],'SAME'),b_conv3),name=scope)
		print_activation(h_conv3)
		parameters+=[W_conv3,b_conv3]
	"""第四个卷积层和池化层"""
	with tf.name_scope('conv4')as scope:
		W_conv4=weight_variables([3,3,384,256],tf.float32,1e-1,'weights')
		b_conv4=bias_variables(0.0,[256],tf.float32,True,'bias')
		h_conv4=tf.nn.relu(tf.nn.bias_add(conv2d(h_conv3,W_conv4,[1,1,1,1],'SAME'),b_conv4),name=scope)
		print_activation(h_conv4)
		parameters+=[W_conv4,b_conv4]
	"""第五个卷积层和池化层"""
	with tf.name_scope('conv5')as scope:
		W_conv5=weight_variables([3,3,256,256],tf.float32,1e-1,'weights')
		b_conv5=bias_variables(0.0,[256],tf.float32,True,'bias')
		h_conv5=tf.nn.relu(tf.nn.bias_add(conv2d(h_conv4,W_conv5,[1,1,1,1],'SAME'),b_conv5),name=scope)
		print_activation(h_conv5)
		parameters+=[W_conv5,b_conv5]
	"""池化层"""
	h_pool5=max_pool_2x2(h_conv5,[1,3,3,1],[1,2,2,1],'VALID','pool5')
	print_activation(h_pool5)
	return h_pool5,parameters
"""5.评估每轮AlexNet时间"""
def cal_AlexNet_time(session,target,info_string):
	first_steps=10
	total_time=0.0
	total_time_squared=0.0
	for i in range(batch_nums+first_steps):
		start_time=time.time()
		_=session.run([target])
		duration=time.time()-start_time
		if i>=first_steps and not i%10:
			print('%s :step %d, duration= %.3f'%(datetime.now(),(i-first_steps),duration))
			total_time+=duration
			total_time_squared+=duration*duration
	sec_pre_bat=total_time/batch_nums
	std_pre_bat=math.sqrt(total_time_squared/batch_nums-math.pow(sec_pre_bat,2))
	print('%s: %s across %d steps,%.3f +/- %.3f seconds/bacth'%(datetime.now(),
		info_string,batch_nums,sec_pre_bat,std_pre_bat))
"""6.主函数"""
def main_run():
	with tf.Graph().as_default():
		image_size=224
		images=tf.Variable(tf.random_normal([batch_size,image_size,image_size,3],dtype=tf.float32,stddev=1e-1))
		pool5,parameters=inference(images)
		init=tf.global_variables_initializer()
		sess=tf.Session()
		sess.run(init)
		cal_AlexNet_time(sess,pool5,'Forward')
		obj=tf.nn.l2_loss(pool5)
		#运用梯度下降
		grad=tf.gradients(obj,parameters)
		cal_AlexNet_time(sess,grad,'Forward-Backward')



