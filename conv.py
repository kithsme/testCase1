import tensorflow as tf
import numpy as np
import preproc
import time
import random

BATCH_SIZE = 2500
CONV1_OUT_CH_NUM = 64
CONV2_OUT_CH_NUM = 128
CONV3_OUT_CH_NUM = 256
CONV4_OUT_CH_NUM = 512
FULLY_CONNECTED_NUM = 1024
DROP_OUT_PROB = 0.5
TRANING_SET_RATE = 0.7
ITERATION = 30

rgb_x, rgb_y = preproc.preproc(rgb=1)
rgb_x_train = rgb_x[:int(len(rgb_y)*TRANING_SET_RATE)]
rgb_x_test = rgb_x[int(len(rgb_y)*TRANING_SET_RATE):]
rgb_y_train = rgb_y[:int(len(rgb_y)*TRANING_SET_RATE)]
rgb_y_test = rgb_y[int(len(rgb_y)*TRANING_SET_RATE):]
x_t,y_t = [],[]
x_f,y_f = [],[]

total_y_true = len(list(filter(lambda y: y==[1,0], rgb_y)))
total_y_false = len(rgb_y) - total_y_true

msgLst = []
msgLst.append('Configurations:')
msgLst.append('\tpreproc:')
msgLst.append('\t\tMIN_LONG = {0}'.format(preproc.MIN_LONG))
msgLst.append('\t\tMAX_LONG = {0}'.format(preproc.MAX_LONG))
msgLst.append('\t\tMIN_LAT = {0}'.format(preproc.MIN_LAT))
msgLst.append('\t\tMAX_LAT = {0}'.format(preproc.MAX_LAT))
msgLst.append('\t\tGRAY_SCALE = {0}'.format(preproc.GRAY_SCALE))
msgLst.append('\t\tSTEP = {0}'.format(preproc.STEP))
msgLst.append('\t\tTIME_WINDOW = {0}'.format(preproc.TIME_WINDOW))
msgLst.append('\tconv:')
msgLst.append('\t\tBATCH_SIZE = {0}'.format(BATCH_SIZE))
msgLst.append('\t\tCONV1_OUT_CH_NUM = {0}'.format(CONV1_OUT_CH_NUM))
msgLst.append('\t\tCONV2_OUT_CH_NUM = {0}'.format(CONV2_OUT_CH_NUM))
msgLst.append('\t\tCONV3_OUT_CH_NUM = {0}'.format(CONV3_OUT_CH_NUM))
msgLst.append('\t\tCONV4_OUT_CH_NUM = {0}'.format(CONV4_OUT_CH_NUM))
msgLst.append('\t\tFULLY_CONNECTED_NUM = {0}'.format(FULLY_CONNECTED_NUM))
msgLst.append('\t\tDROP_OUT_PROB = {0}'.format(DROP_OUT_PROB))
msgLst.append('\tTRANING_SET_RATE = {0}'.format(TRANING_SET_RATE))
msgLst.append('\tITERATION = {0}'.format(ITERATION))


for a in filter(lambda enu: enu[1]==[1,0], enumerate(rgb_y_test)):
    x_t.append(rgb_x_test[a[0]])
    y_t.append([1,0])

for a in filter(lambda enu: enu[1]==[0,1], enumerate(rgb_y_test)):
    x_f.append(rgb_x_test[a[0]])
    y_f.append([0,1])


print('\nData Information:')
print('\tTOTAL SET SIZE = {0}'.format(len(rgb_x)))
print('\t\tTOTAL TRUE/FALSE COUNT = {0}/{1}'.format(total_y_true, total_y_false))
print('\tTRAINING SET SIZE = {0}'.format(len(rgb_x_train)))
print('\tTEST SET SIZE = {0}'.format(len(rgb_x_test)))
print('\t\tTOTAL TRUE/FALSE COUNT IN TEST SET = {0}/{1}'.format(len(x_t), len(x_f)))

msgLst.append('\nData Information:')
msgLst.append('\tTOTAL SET SIZE = {0}'.format(len(rgb_x)))
msgLst.append('\t\tTOTAL TRUE/FALSE COUNT = {0}/{1}'.format(total_y_true, total_y_false))
msgLst.append('\tTRAINING SET SIZE = {0}'.format(len(rgb_x_train)))
msgLst.append('\tTEST SET SIZE = {0}'.format(len(rgb_x_test)))
msgLst.append('\t\tTOTAL TRUE/FALSE COUNT IN TEST SET = {0}/{1}'.format(len(x_t), len(x_f)))

it = len(rgb_x_train)//BATCH_SIZE

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, preproc.STEP*preproc.STEP*3])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

def shift(rgb_x, rgb_y):
    
    rgb_x_firstBat = rgb_x[:BATCH_SIZE]
    rgb_y_firstBat = rgb_y[:BATCH_SIZE]

    rgb_x = rgb_x[BATCH_SIZE:] + rgb_x_firstBat
    rgb_y = rgb_y[BATCH_SIZE:] + rgb_y_firstBat 

    return rgb_x, rgb_y

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

W_conv1 = weight_variable([5,5,3,CONV1_OUT_CH_NUM])
b_conv1 = bias_variable([CONV1_OUT_CH_NUM])

x_image = tf.reshape(x, [-1,preproc.STEP,preproc.STEP,3])
pool = 0
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
print(h_conv1)
h_pool1 = max_pool_2x2(h_conv1)
print(h_pool1)
pool += 1

W_conv2 = weight_variable([5,5,CONV1_OUT_CH_NUM,CONV2_OUT_CH_NUM])
b_conv2 = bias_variable([CONV2_OUT_CH_NUM])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
print(h_conv2)
h_pool2 = max_pool_2x2(h_conv2)
print(h_pool2)
pool += 1

W_conv3 = weight_variable([5,5,CONV2_OUT_CH_NUM,CONV3_OUT_CH_NUM])
b_conv3 = bias_variable([CONV3_OUT_CH_NUM])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
print(h_conv3)
h_pool3 = max_pool_2x2(h_conv3)
print(h_pool3)
pool += 1

W_conv4 = weight_variable([5,5,CONV3_OUT_CH_NUM, CONV4_OUT_CH_NUM])
b_conv4 = bias_variable([CONV4_OUT_CH_NUM])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
print(h_conv4)
h_pool4 = max_pool_2x2(h_conv4)
print(h_pool4)
pool += 1

fc_num = int(preproc.STEP/(2**pool))
fc_num = 1
W_fc1 = weight_variable([fc_num*fc_num*CONV3_OUT_CH_NUM, FULLY_CONNECTED_NUM])
b_fc1 = bias_variable([FULLY_CONNECTED_NUM])

h_pool4_flat = tf.reshape(h_conv3, [-1, fc_num*fc_num*CONV3_OUT_CH_NUM])
print(h_pool4_flat)
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1)+b_fc1)
print(h_fc1)

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

print('\n-------------------------------------------------')
msgLst.append('\n-------------------------------------------------')
start_time = time.time()
for j in range(ITERATION):
    print('Iteration ({0}/{1}) starts training...'.format(j, ITERATION))
    msgLst.append('Iteration ({0}/{1}) :'.format(j, ITERATION))
    for i in range(it):
        xx = rgb_x_train[BATCH_SIZE*i:BATCH_SIZE*i+BATCH_SIZE]
        yy = rgb_y_train[BATCH_SIZE*i:BATCH_SIZE*i+BATCH_SIZE]
        train_step.run(feed_dict={x: xx, y_: yy, keep_prob:DROP_OUT_PROB})
        print('...{0}...'.format(i+1))

    rgb_x_train, rgb_y_train = shift(rgb_x_train, rgb_y_train)




f_train_acc = sess.run(accuracy, feed_dict={x:rgb_x_train, y_:rgb_y_train, keep_prob:1.0})
f_test_acc = sess.run(accuracy, feed_dict={x:rgb_x_test, y_:rgb_y_test, keep_prob:1.0})
f_true_only_acc = sess.run(accuracy, feed_dict={x:x_t, y_:y_t, keep_prob:1.0})
f_false_only_acc = sess.run(accuracy, feed_dict={x:x_f, y_:y_f, keep_prob:1.0})
print('\n-------------------------------------------------')
print('Final conv: training accuracy: %g'%f_train_acc)
print('Final conv: test accuracy: %g'%f_test_acc)
print('Final conv: true only set accuracy: %g'%f_true_only_acc)
print('Final conv: false only set accuracy: %g'%f_false_only_acc)
print('Total spent time: {0} (sec)'.format(time.time()-start_time))
msgLst.append('\n-------------------------------------------------')
msgLst.append('Final conv: training accuracy: %g'%f_train_acc)
msgLst.append('Final conv: test accuracy: %g'%f_test_acc)
msgLst.append('Final conv: true only set accuracy: %g'%f_true_only_acc)
msgLst.append('Final conv: false only set accuracy: %g'%f_false_only_acc)
msgLst.append('Total spent time: {0} (sec)'.format(time.time()-start_time))

file_name_suffix = time.strftime('%Y-%m-%d_%H-%M-%S')
log_file = open('./result/log_{0}.txt'.format(file_name_suffix), 'w')
for l in msgLst:
    log_file.write(l+'\n')

log_file.close()
