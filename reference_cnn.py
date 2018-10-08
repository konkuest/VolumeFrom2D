'''
Reference File for building CNN
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True, reshape=False)

X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
Y_Label = tf.placeholder(tf.float32, shape=[None,10])

#커널(필터)은 (4x4x1)의 필터를 4장 사용하기 위해 shape을 설정하고, tf.truncated_normal()을 통해 초기화 (initialize)
#이 Kernel이 학습이 될 변수 (Weight 값)을 의미함 -> 계속 업데이트될 변수이기 때문에 tf.Variable로 지정
Kernel1 = tf.Variable(tf.truncated_normal(shape=[4,4,1,4], stddev=0.1))

#이미지와 커널을 Convolution한 이후 같은 사이즈 만큼 더해주기 위핸 변수를 의미, 커널이 4장 있었기 때문에 똑같이 shape=[4]
Bias1 = tf.Variable(tf.truncated_normal(shape=[4], stddev=0.1))

#Convolution 연산 시작
#X라는 이미지에 Kernel1 값을 그대로 컨볼루젼 곱을 실행하고, 패딩 사이즈는 출력 사이즈에 맞게 알아서 세팅됨
Conv1 = tf.nn.conv2d(X, Kernel1, strides=[1,1,1,1], padding='SAME') + Bias1

#ReLU Activation Function
Activation1 = tf.nn.relu(Conv1)

#Max_Pooling 사용
Pool1 = tf.nn.max_pool(Activation1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#첫 번째 Convolution Layer와 유사하게 기능
Kernel2 = tf.Variable(tf.truncated_normal(shape=[4,4,4,8], stddev=0.1))
Bias2 = tf.Variable(tf.truncated_normal(shape=[8], stddev=0.1))
Conv2 = tf.nn.conv2d(Pool1, Kernel2, strides=[1,1,1,1], padding='SAME') + Bias2
Activation2 = tf.nn.relu(Conv2)
Pool2 = tf.nn.max_pool(Activation2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

W1 = tf.Variable(tf.truncated_normal(shape=[8*7*7,10]))
B1 = tf.Variable(tf.truncated_normal(shape=[10]))
Pool2_flat = tf.reshape(Pool2, [-1, 8*7*7])
OutputLayer = tf.matmul(Pool2_flat, W1) + B1

Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_Label, logits=OutputLayer))
train_step = tf.train.AdamOptimizer(0.005).minimize(Loss)

correct_prediction = tf.equal(tf.argmax(OutputLayer, 1), tf.argmax(Y_Label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    print("Start....")
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        trainingData, Y = mnist.train.next_batch(64)
        sess.run(train_step, feed_dict={X: trainingData, Y_Label: Y})
        if i % 100 :
            print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y_Label: mnist.test.labels}))