import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_epochs=100 # 每个循环周期
total_series_length = 50000 # 数据总长度
truncated_backprob_length = 14 # 截断反传长度
state_size = 4 #状态数量
num_classes = 2
echo_step = 3 # 在echo_step个时间步后得到输入的回声
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprob_length

def generateData():
    # 从[0,1]中生成total_series_length长度的数组，0和1概率各为0.5
    x = np.array(np.random.choice(2,total_series_length,p=[0.5,0.5]))
    y = np.roll(x,echo_step)
    y[0:echo_step] = 0
    x = x.reshape((batch_size,-1))
    y = y.reshape((batch_size,-1))
    return (x,y)

X = tf.placeholder(tf.float32,[batch_size,truncated_backprob_length])
Y = tf.placeholder(tf.float32,[batch_size,truncated_backprob_length])
init_state = tf.placeholder(tf.float32,[batch_size,state_size])

W = tf.Variable(np.random.rand(state_size+1,state_size),dtype=tf.float32)
b = tf.Variable(np.zeros((1,state_size)),dtype=tf.float32)
W2 = tf.Variable(np.random.rand(state_size,num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)),dtype=tf.float32)