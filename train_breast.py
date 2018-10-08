#-*- coding: utf-8 -*-

'''
Two Issues not solved
1) 이미지 파일을 리사이징 해서 숫자 데이터로 변경하기
2) 정확하게 컨볼루션 짜기
'''

import tensorflow as tf
import random

# 데이터 정의
X = tf.placeholder(tf.float32, [None, 784]) #이미지 픽셀 값 12,192,768개 (4032 x 3024)
X_img = tf.reshape(X, [-1, 28, 28, 3]) #-1로 알아서 맞추고, 200x200 사이즈, 색깔은 3가지 (RGB)로 구분
Y = tf.placeholder(tf.float32, [None, 10]) #총 8개의 Breast Sample로 구분할 것임

# Convolution 1
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01)) #필터 사이즈 16x16, 색은 3가지, 필터개수 32개
L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME') #Stride 4로 해서 모두 Convolution
L1 = tf.nn.relu(L1) #ReLU 통과
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') #Max Pooling, 커널 사이즈 2x2, stride 2x2로 이동

# Convolution 2
W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01)) #필터개수 64개로 증가
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# Spread
L2 = tf.reshape(L2, [-1, 7 * 7 * 64, 10])

# Hypothesis
W3 = tf.get_variable("W2", shape=[7 * 7 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L2, W3) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

# Initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

'''
# Train
print("###### Learning started ######")

training_epochs = 15
batch_size = 10

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int()
'''