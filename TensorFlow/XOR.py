#Goal: Simple feed forward network
import tensorflow as tf
import numpy as np


def dataGenerator(n = 100):
    X = np.random.randint(0, 2, (n, 2))
    y = np.array([int(x[0] != x[1]) for x in X]).reshape((n,1))
    return X,y


X, y = dataGenerator()

x_ = tf.placeholder(tf.float32, shape=[None, 2], name="X")
y_ = tf.placeholder(tf.float32, shape=[None, 1], name="y")
W_1 = tf.Variable(tf.random_uniform([X.shape[1], 2], -1, 1), name="W_1")
W_output = tf.Variable(tf.random_uniform([2, 1], -1, 1), name="W_output")
b_1 = tf.Variable(tf.zeros([1]), name="b_1")
b_output = tf.Variable(tf.zeros([1]), name="b_output")

hidden_layer = tf.sigmoid(tf.matmul(x_, W_1) + b_1)
output_layer = tf.sigmoid(tf.matmul(hidden_layer, W_output) + b_output)

cost = tf.reduce_mean(tf.square(output_layer - y_))
train_step = tf.train.AdamOptimizer().minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
epoch = 20000
for i in range(epoch):
    sess.run(train_step, feed_dict={x_: X, y_: y})
    if i % 1000 == 0:
        print("Mean Squared Error: ", sess.run(cost, feed_dict={x_: X, y_: y}))


y_pred = sess.run(output_layer,feed_dict={x_: [[0, 0], [0, 1], [1, 0], [1, 1]]})
