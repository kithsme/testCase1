import tensorflow as tf
import numpy as np
import preproc

BATCH_SIZE = 100
CONV_OUT_CH_NUM = 32
FULLY_CONNECTED_NUM = 1024
DROP_OUT_PROB = 0.7

rgb_x, rgb_y, x_t, x_f = preproc.preproc(rgb=1)
rgb_x_train = rgb_x[:int(len(rgb_y)*0.75)]
rgb_x_test = rgb_x[int(len(rgb_y)*0.75):]
rgb_y_train = rgb_y[:int(len(rgb_y)*0.75)]
rgb_y_test = rgb_y[int(len(rgb_y)*0.75):]
y_t = []
y_f = []
for _ in x_t:
    y_t.append([1,0])

for _ in x_f:
    y_f.append([0,1])

it = len(rgb_x_train)//BATCH_SIZE

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, preproc.STEP*preproc.STEP*3])
y_ = tf.placeholder(tf.float32, shape=[None, 2])
'''
W = tf.Variable(tf.zeros([32*32*3,2]))
b = tf.Variable(tf.zeros([2]))

sess.run(tf.global_variables_initializer())

y = tf.matmul(x,W)+b

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(1000):
    xx = rgb_x_train[50*i:50*i+50]
    yy = rgb_y_train[50*i:50*i+50]
    train_step.run(feed_dict={x: xx, y_: yy})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('one layer test accuracy: %g'%accuracy.eval(feed_dict={x:rgb_x_test, y_:rgb_y_test}))
print('one layer: true set accuracy: %g'%accuracy.eval(feed_dict={x:x_t, y_:y_t}))

'''
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # small noise for symmetry breaking
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape) # prevent dead neuron 
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

W_conv1 = weight_variable([5,5,3,CONV_OUT_CH_NUM])
b_conv1 = bias_variable([CONV_OUT_CH_NUM])

x_image = tf.reshape(x, [-1,preproc.STEP,preproc.STEP,3])
pool = 0
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
print(h_conv1)
h_pool1 = max_pool_2x2(h_conv1)
print(h_pool1)
pool += 1

W_conv2 = weight_variable([5,5,CONV_OUT_CH_NUM,CONV_OUT_CH_NUM])
b_conv2 = bias_variable([CONV_OUT_CH_NUM])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
print(h_conv2)
h_pool2 = max_pool_2x2(h_conv2)
print(h_pool2)
pool += 1

W_conv3 = weight_variable([5,5,CONV_OUT_CH_NUM,CONV_OUT_CH_NUM])
b_conv3 = bias_variable([CONV_OUT_CH_NUM])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
print(h_conv3)
h_pool3 = max_pool_2x2(h_conv3)
print(h_pool3)
pool += 1

W_conv4 = weight_variable([5,5,CONV_OUT_CH_NUM, CONV_OUT_CH_NUM])
b_conv4 = bias_variable([CONV_OUT_CH_NUM])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
print(h_conv4)
h_pool4 = max_pool_2x2(h_conv4)
print(h_pool4)
pool += 1

fc_num = int(preproc.STEP/(2**pool))
W_fc1 = weight_variable([fc_num*fc_num*CONV_OUT_CH_NUM, FULLY_CONNECTED_NUM])
b_fc1 = bias_variable([FULLY_CONNECTED_NUM])

h_pool4_flat = tf.reshape(h_pool4, [-1, fc_num*fc_num*CONV_OUT_CH_NUM])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1)+b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([FULLY_CONNECTED_NUM,2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
sess.run(tf.global_variables_initializer())

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for j in range(10):
    for i in range(it):
        if i%100==0: 
            print('Batch ({0}/{1}) starts training...'.format((it*j+i+1), it*10))
        xx = rgb_x_train[BATCH_SIZE*i:BATCH_SIZE*i+BATCH_SIZE]
        yy = rgb_y_train[BATCH_SIZE*i:BATCH_SIZE*i+BATCH_SIZE]
        train_step.run(feed_dict={x: xx, y_: yy, keep_prob:DROP_OUT_PROB})

print('\n-------------------------------------------------')
print('conv: test accuracy: %g'%accuracy.eval(feed_dict={x:rgb_x_test, y_:rgb_y_test, keep_prob:1.0}))
print('conv: true only set accuracy: %g'%accuracy.eval(feed_dict={x:x_t, y_:y_t, keep_prob:1.0}))
#print('conv: false only set accuracy: %g'%accuracy.eval(feed_dict={x:x_f, y_:y_f, keep_prob:1.0}))

