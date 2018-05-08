# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf

# 下载并载入 MNIST 手写数字库(55000*28*28)55000张训练图像
from tensorflow.examples.tutorials.mnist import imput_data

mnist = input_data.read_data_sets('mnist_data', one_hot=True)

# one_hot 独热码的编码（encoding）形式
# 0,1,2,3,4,5,6,7,8,9十位数字
# 0 ：1000000000
# 1 ：0100000000

#placeholder 占位符 None 表示张量（Tensor）的第一个维度可以是任何长度
input_x = tf.placeholder(tf.float32, [None, 28*28])/255
output_y = tf.placeholder(tf.float32,[None, 10])# 输出
#该新矩阵的维数为（a，28，28，1），其中-1表示a由实际情况来定。
input_x_images = tf.reshape(input_x, [-1, 28, 28, 1])

# 从Test(测试)数据集里选取3000个手写数字的图片和对应的标签 

test_x = mnist.test.images[:3000] #图片
test_y = mnist.test.labels[:3000] #标签

# 构建我们的卷积神经网络
# 第一层卷积

conv1 = tf.layers.conv2d(
        inputs=input_x_images,# 形状[28 ,28, 1]
        filters=32,         # 32个过滤器，输出的深度（depth、channel)是32
        kernel_size=[5,5],  # 过滤器在二维的大小是（5×5）
        strides=1,          # 步长是1
        padding='same',     # same表示输出的大小不变
        activation=tf.nn.relu # 激活函数
        
        ) #形状[28, 28, 32] 

# 第一层池化（亚采样）
pool1 = tf.layers.max_pooling2d(
        inputs=conv1, # 形状[28,28,32]
        pool_size=[2, 2], # 过滤器
        strides=2,
        ) # 形状[14, 14, 32]
        
# 第二层卷积

conv2 = tf.layers.conv2d(
        inputs=pool1,# 形状[14 ,14, 32]
        filters=64,         # 64个过滤器，输出的深度（depth、channel)是32
        kernel_size=[5,5],  # 过滤器在二维的大小是（5×5）
        strides=1,          # 步长是1
        padding='same',     # same表示输出的大小不变
        activation=tf.nn.relu # 激活函数
        
        ) #形状[14, 14, 64]

# 第二层池化
pool2 = tf.layers.max_pooling2d(
        inputs=conv2, # 形状[28,28,32]
        pool_size=[2, 2], # 过滤器
        strides=2,
        ) # 形状[7, 7, 64]



# 平坦化
flat = tf.reshape(pool2, [-1,7*7*64]) #形状[7*7*64]

# 1024 个神经元全链接层
dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)

# Dropout : 丢弃 50%, rate=0.5
dropout = tf.layers.dropout(inputs=dense, rate=0.5)

# 10个神经元的全链接层，这里不用激活函数来做非线性化
logits = tf.layers.dense(unputs=droput, units=10)# 输出 形状[1, 1, 10]

# 计算误差loss(Cross entropy)

loss = de.losses.softmax_cross_entropy(onehot_labels=output_y, logits=logits)

# Adam 优化器最小化误差， 学习率0.001

train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize()

# 精度。计算预测值和实际标签的匹配程度
# 返回(accuracy, update_op), 会创建两个局部变量

accuracy = tf.metrics.accuracy(
        labels=tf.argmax(output_y, axis=1),
        predictions=tf.argmax(logits,axis=1))[1]

sess = tf.Session()

# 初始化变量：因为含有局部变量(全局和局部)
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

sess.run(init)


for i in range(20000):
    batch = mnist.train.next_batch(50)  # 从Train(训练）数据集里读取
    train_loss,train_op_ = sess.run([loss, train_op],{input_x:batch[0], output_y:batch[1]})
    if i %100==0:
       test_accuracy = sess.run(accuracy, {{input_x:test_x, output_y:test_y})
    print("Step=%d, Train loss=%.4f,[Test accuracy=%.2f]")\
        %(i,train_loss,test_accuracy)

# 测试：打印20个预测值和真实值的对
test_output = sess.run(logits,{input_x: test_x[:20]})
inferenced_y =np.argmax(text_output,1)
print(inferenced_y, 'Inferenced numbers') #推测的数字
print(np.argmax(test_y[:20],1), 'Real numbers') # 真实的数字    




