  # -*- coding: utf-8 -*-
"""
project:Tensorflow实现VGGNet卷积神经
@Author:Gene
@Github:https://github.com/Gene20/DeepLearning
@Email:GeneWithyou@gamil.com
@Website:www.gene20.top
"""  
"""
VGGNet16网络结构：
Input-->Conv*2+Pool-->Conv*2+Pool-->Conv*3+Pool-->Conv*3+Pool
-->Conv*3+Pool-->FC*2-->FC+SoftMax-->Output
"""
from datetime import datetime 
import math
import time
import tensorflow as tf
"""创建卷积层并存入本层参数"""
def conv2d_op(x,name,shape,strides,pams):
	with tf.name_scope(name) as scope:
		W_conv=tf.get_variable(scope+'w',shape=shape,dtype=tf.float32,
			initializer=tf.contrib.layers.xavier_initializer())
		b_conv=tf.Variable(tf.constant(0.0,shape=[shape[-1]],dtype=tf.float32),trainable=True,name='b')
		conv=tf.conv2d(x,W_conv,strides=strides,padding='SAME')
		h_conv=tf.nn.relu(tf.bias_add(conv,b_conv),name=scope)
		pams+=[W_conv,b_conv]
		return h_conv
"""创建全连接层"""
def fc_op(x,name,shape,pams):
	with tf.name_scope(name) as scope:
		W_fc=tf.get_variable(scope+'w',shape=shape,dtype=tf.float32,
			initializer=tf.contrib.layers.xavier_initializer())
		b_fc=tf.Variable(tf.constant(0.1,shape=[shape[-1]]),dtype=tf.float32,trainable=True,name='b')
		h_fc=tf.nn.relu(tf.matmul(x,W_fc)+b_fc,name=scope)
		pams+=[W_fc,b_fc]
		return h_fc
"""创建最大池化层"""
def max_pool_op(x,name,ksize,strides):
	return tf.nn.max_pool(x,ksize=ksize,strides=strides,padding='SAME',name=name)

"""所有的卷积层和池化层"""
def inference_op(x,keep_prob):
	pams=[]
	first_in=x.get_shape()[-1].value
"""第一个卷积层和池化层"""
	conv1_1=conv2d_op(x,'conv1_1',[3,3,first_in,64],[1,1,1,1],pams)
	conv1_2=conv2d_op(conv1_1,'conv1_2',[3,3,64,64],[1,1,1,1],pams)
	pool1=max_pool_op(conv1_2,'pool1',[1,2,2,1],[1,2,2,1])
"""第二个卷积层和池化层"""
	conv2_1=conv2d_op(pool1,'conv2_1',[3,3,64,128],[1,1,1,1],pams)
	conv2_2=conv2d_op(conv2_1,'conv2_2',[3,3,128,128],[1,1,1,1],pams)
	pool2=max_pool_op(conv2_2,'pool2',[1,2,2,1],[1,2,2,1])
"""第三个卷积层和池化层"""
	conv3_1=conv2d_op(pool2,'conv3_1',[3,3,128,256],[1,1,1,1],pams)
	conv3_2=conv2d_op(conv3_1,'conv3_2',[3,3,256,256],[1,1,1,1],pams)
	conv3_3=conv2d_op(conv3_2,'conv3_3',[3,3,256,256],[1,1,1,1],pams)
	pool3=max_pool_op(conv3_3,'poo3',[1,2,2,1],[1,2,2,1])
"""第四个卷积层和池化层"""
	conv4_1=conv2d_op(pool3,'conv4_1',[3,3,256,512],[1,1,1,1],pams)
	conv4_2=conv2d_op(conv4_1,'conv4_2',[3,3,512,512],[1,1,1,1],pams)
	conv4_3=conv2d_op(conv4_2,'conv4_3',[3,3,512,512],[1,1,1,1],pams)
	pool4=max_pool_op(conv4_3,'pool4',[1,2,2,1],[1,2,2,1])
"""第五个卷积层和池化层"""
	conv5_1=conv2d_op(pool4,'conv5_1',[3,3,512,512],[1,1,1,1],pams)
	conv5_2=conv2d_op(conv5_1,'conv5_2',[3,3,512,512],[1,1,1,1],pams)
	conv5_3=conv2d_op(conv5_2,'conv5_3',[3,3,512,512],[1,1,1,1],pams)
	pool5=max_pool_op(conv5_3,'pool5',[1,2,2,1],[1,2,2,1])
"""输入结果扁平化flatten"""
	sp=pool5.get_shape()
	dims=sp[1].value*sp[2].value*sp[3].value
	#将样本化为7*7*512=25088的一维向量
	rsp=tf.reshape(pool5,[-1,dims],name='rsp')
"""全连接层+DropOut层"""
	fc6=fc_op(rsp,'fc6',[dims,4096],pams)
	fc6_dp=tf.nn.dropout(fc6,keep_prob,name='fc6_dp')
	fc7=fc_op(fc6_dp,'fc7',[4096,4096],pams)
	fc7_dp=tf.nn.dropout(fc7,keep_prob,name='fc7_dp')
"""1000个输出节点的全连接层(softmax函数处理)"""
	fc8=fc_op(fc7_dp,'fc8',[4096,1000],pams)
	result_sfmx=tf.nn.softmax(fc8)
	results=tf.argmax(result_sfmx,1)
	return results,result_sfmx,fc8,pams
"""评测函数"""
def test_run(session,target,feed,info_string):
	first_steps=10#给程序热身
	total_time=0.0
	total_time_sq=0.0
	for i in (batch_nums+first_steps):
		start_time=time.time()
		_=session.run(target,feed_dict=feed)
		duration=time.time()-start_time
		if i>=first_steps and not i%10:
			print('%s: steps:%d ,duration:%.3f'%(datetime.now(),i,duration))
			total_time+=duration
			total_time_sq+=duration*duration
	sec_per_batch=total_time/batch_nums
	std_per_batch=math.sqrt(total_time_sqt/batch_nums-math.pow(sec_per_batch,2))
	print('%s:%s across %d steps,%.3f +/- %.3f seconds/batch'%(datetime.now(),
		info_string,batch_nums,sec_per_batch,std_per_batch))
"""主函数"""
def main_run():
	with tf.Graph().as_default():
		image_size=224
		iamges=tf.Variable(tf.random_normal([batch_size,image_size,image_size,3],dtype=tf.float32,stddev=1e-1))
		keep_prob=tf.placeholder(tf.float32)
		results,result_sfmx,fc8,pams=inference_op(images,keep_prob)
		init=tf.global_variables_initializer()
		sess=tf.Session()
		sess.run(init)
		test_run(sess,results,{keep_prob:1.0},'Forward')
		obj=tf.nn.l2_loss(fc8)
		grad=tf.gradients(obj,pams)
		test_run(sess,grad,{keep_prob:0.5},'Backward')
"""进行评估"""
batch_nums=32
batch_size=100
main_run()
