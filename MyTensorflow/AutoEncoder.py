# -*- coding: utf-8 -*-
"""
project:4.2TensorFlow实现自编码器
@Author:Gene
@Github:https://github.com/Gene20/DeepLearning
@Email:GeneWithyou@gamil.com
@Website:www.gene20.top
"""
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

"""1.实现标准的均匀分布Xaiver初始化器"""
def xaiver_init(fan_in,fan_out,constant=1):
    low=-constant*np.sqrt(6.0/(fan_in+fan_out))
    high=constant*np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in,fan_out),
                             minval=low,maxval=high,
                             dtype=tf.float32)

"""2.定义去噪自编码的类"""
class AutoEncoder(object):
    def __init__(self,n_input,n_hidden,transfer_function=tf.nn.softplus,optimizer=tf.train.AdamOptimizer(),scale=0.1):
        self.n_input=n_input
        self.n_hidden=n_hidden
        self.transfer=transfer_function
        self.scale=tf.placeholder(tf.float32)
        self.training_scale=scale
        network_weights=self._initialize_dweights()
        self.weights=network_weights
        self.x=tf.placeholder(tf.float32,[None,self.n_input])
        self.hidden=self.transfer(tf.add(tf.matmul(self.x+scale*tf.random_normal((n_input)),
                                                   self.weights['w1']),self.weights['b1']))
        self.reconstruction=tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])

#定义自编码器的损失函数
        self.cost=0.5*tf.reduce_sum(tf.pow(tf.substract(self.reconstruction,self.x),2.0))
        self.optimizer=optimizer.minimize(self.cost)
        init=tf.global_variables_initializer()
        self.sess=tf.Session()
        self.sess.run(init)
"""3.参数初始化"""
    def _initialize_weights(self):
        all_weights=dict()
        all_weights['w1']=tf.Variable(xaiver_init(self.n_input,self.n_hidden))
        all_weights['b1']=tf.Variable(tf.zeros([self.n_hidden],dtype=float32))
        all_weights['w2']=tf.Variable(tf.zeros([self.n_hidden,self.n_input],dtype=float32))
        all_weights['b2']=tf.Variable(tf.zeros([self.n_input],dtype=float32))
        return all_weights
    
"""4.执行一步训练并返回损失"""
    def partial_fit(self,X):
        cost,opt=self.sess.run((self.cost,self.optimizer),
                               feed_dict={self.x:X,self.scale:self.training_scale})
        return cost

"""5.计算测试的损失""" 
    def calc_total_cost(self,X):
        return self.sess.run(self.cost,feed_dict={self.x:X,self.scale:self.training_scale})

"""6.返回自编码器隐含层的输出结果"""
    def transform(self,X):
        return self.sess.run(self.hidden,feed_dict={self.x:X,self.scale:self.training_scale})

"""7.将高阶特征复原为原始数据"""
    def generate(self,hidden=None):
        if hidden is None:
            hidden=np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction,feed_dict={self.hidden:hidden})

"""8.整合函数"""
    def reconstruct(self,X):
        return self.sess.run(self.reconstruction,feed_dict={self.x:X,
                                                        self.scale:self.training_scale})

"""9.获取  隐含层权重w1"""
    def getWeights(self):
        return self.sess.run(self.weights['w1'])
"""10.获取隐含层偏置b1"""
    def getBiases(self):
        return self.sess.run(self.weights['b1'])

"""11.进行训练并测试"""
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

"""12.对数据进行标准化处理"""
def standard_scale(X_train,X_test):
    preces=prep.StandardScaler().fit(X_train)
    X_train=preces.transform(X_train)
    X_test=preces.transform(X_test)
    return X_train,X_test

"""13.随机获取block的函数"""
def get_random_block_from_data(data,batch_size):
    start=np.random.randint(0,len(data)-batch_size)
    return data[start:(start+batch_size)]

X_train,X_test=standard_scale(mnist.train.images,mnist.test.images)
n_samples=int(mnist.train.num_examples)#总训练样本数
training_epochs=20#最大训练轮数
batch_size=128#批量训练的个数
display_step=1

#创建AGN自编码器的实例
autoEncoder=AutoEncoder(n_input=784,
                        n_hidden=200,
                        transfer_function=tf.nn.softplus,
                        optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                        scale=0.01)

"""14.开始训练过程"""
for epoch in range(training_epochs):
    avg_cost=0
    total_batch=int(n_samples/batch_size)
    for i in range(total_batch):
        batch_xs=get_random_block_from_data(X_train,batch_size)
        cost=autoEncoder.partial_fit(batch_xs)
        avg_cost+=cost/n_samples*batch_size
    if epoch%display_step==0:
        print('Epoch:','%04d'%(epoch+1),'cost=',
              '{:9f}'.format(avg_cost))
              
 
    
