import tensorflow as tf
import conf
import numpy as np
import preproc
import genImage
import time
import string
import random
import os
import matplotlib.pyplot as plt
import math
from PIL import Image
import pandas as pd
import csv

global config

ANUM=166



config = conf.conf()

def read_coords(fn):
    orderdata = []
    f = open(fn)
    csvReader = csv.reader(f)

    cnt = 0
    for row in csvReader:
        
        xy = []
        xy.append(float(row[0]))
        xy.append(float(row[1]))
        xy.append(float(row[2]))
        xy.append(float(row[3]))
        xy.append(float(row[4]))
        xy.append(float(row[5]))
        xy.append(float(row[6]))
        xy.append(float(row[7]))
        xy.append(int(row[8]))
        xy.append(str(row[9]))
        xy.append(str(row[10]))

        orderdata.append(xy)


    return orderdata

def get_samples():
    coord_xy = read_coords('coord_xy.csv')
    log_preproc_conf()
    con = genImage.getConfig(config.CONFIG_ID)
    #gm = genImage.draw_map(train_xy,test_xy,con)
    gm = genImage.draw_map(coord_xy,con)

    random.shuffle(coord_xy)

    train_coord_xy = coord_xy[:int(len(coord_xy)*config.TRANING_SET_RATE)]
    test_coord_xy = coord_xy[int(len(coord_xy)*config.TRANING_SET_RATE):]

    train_x, train_y, _ = genImage.xy_to_rgbArray(train_coord_xy, gm, con)

    test_x, test_y, test_ids = genImage.xy_to_rgbArray(test_coord_xy, gm, con, gen_all=True)

    #test_x, test_y = genImage.xy_to_rgbArray(test_xy, gm, con)

    return train_x, train_y, test_x, test_y, test_ids
    
# def split_training_test(rgb_x, rgb_y):
#     rgb_x_train = rgb_x[:int(len(rgb_y)*config.TRANING_SET_RATE)]
#     rgb_x_test = rgb_x[int(len(rgb_y)*config.TRANING_SET_RATE):]
#     rgb_y_train = rgb_y[:int(len(rgb_y)*config.TRANING_SET_RATE)]
#     rgb_y_test = rgb_y[int(len(rgb_y)*config.TRANING_SET_RATE):]
#     x_t,y_t = [],[]
#     x_f,y_f = [],[]
#     total_y_true = len(list(filter(lambda y: y==[1,0], rgb_y)))
#     total_y_false = len(rgb_y) - total_y_true
#     for a in filter(lambda enu: enu[1]==[1,0], enumerate(rgb_y_test)):
#         x_t.append(rgb_x_test[a[0]])
#         y_t.append([1,0])

#     for a in filter(lambda enu: enu[1]==[0,1], enumerate(rgb_y_test)):
#         x_f.append(rgb_x_test[a[0]])
#         y_f.append([0,1])

#     return rgb_x_train, rgb_y_train, rgb_x_test, rgb_y_test, x_t, y_t, x_f, y_f, total_y_true, total_y_false

def log_preproc_conf():
    if not os.path.exists("./model"):
        os.makedirs("./model")
    if not os.path.exists("./model/{0}".format(config.CONFIG_ID)):
        os.makedirs("./model/{0}".format(config.CONFIG_ID))
    log_file = open('./model/{0}/preproc_config.csv'.format(config.CONFIG_ID), 'w')
    content = [str(config.MIN_LONG), str(config.MAX_LONG), str(config.MIN_LAT), str(config.MAX_LAT), str(config.GRAY_SCALE), str(config.STEP)]
    log_file.write(','.join(content))
    log_file.close()

def log_topology():
    contents = []
    contents.append(','.join([str(x) for x in [config.CONV1_FILTER_WIDTH,config.CONV1_FILTER_HEIGHT,3,config.CONV1_OUT_CH_NUM]]))
    contents.append(','.join([str(x) for x in [config.CONV2_FILTER_WIDTH,config.CONV2_FILTER_HEIGHT,config.CONV1_OUT_CH_NUM,config.CONV2_OUT_CH_NUM]]))
    contents.append(','.join([str(x) for x in [config.CONV3_FILTER_WIDTH,config.CONV3_FILTER_HEIGHT,config.CONV2_OUT_CH_NUM,config.CONV3_OUT_CH_NUM]]))
    contents.append(','.join([str(x) for x in [config.CONV4_FILTER_WIDTH,config.CONV4_FILTER_HEIGHT,config.CONV3_OUT_CH_NUM,config.CONV4_OUT_CH_NUM]]))
    contents.append(','.join([str(x) for x in [config.FC_FILTER_WIDTH*config.FC_FILTER_HEIGHT*config.CONV4_OUT_CH_NUM, config.FULLY_CONNECTED_NUM]]))
    contents.append(','.join([str(x) for x in [config.FULLY_CONNECTED_NUM,2]]))
    
    # assume that './model' path exist
    log_file = open('./model/{0}/topology.csv'.format(config.CONFIG_ID), 'w')
    for a in contents:
        log_file.write(a+'\n')
    log_file.close()

def log_param(sess):
    # assume that '/model' path exists
    W1_val, b1_val, W2_val, b2_val, W3_val, b3_val, W4_val, b4_val, W_fc1_val, b_fc1_val, W_fc2_val, b_fc2_val = sess.run([W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4, W_fc1, b_fc1, W_fc2, b_fc2])
    np.savetxt("./model/{0}/trained_W1.csv".format(config.CONFIG_ID), np.ravel(W1_val), delimiter=",")
    np.savetxt("./model/{0}/trained_b1.csv".format(config.CONFIG_ID), b1_val, delimiter=",")
    np.savetxt("./model/{0}/trained_W2.csv".format(config.CONFIG_ID), np.ravel(W2_val), delimiter=",")
    np.savetxt("./model/{0}/trained_b2.csv".format(config.CONFIG_ID), b2_val, delimiter=",")
    np.savetxt("./model/{0}/trained_W3.csv".format(config.CONFIG_ID), np.ravel(W3_val), delimiter=",")
    np.savetxt("./model/{0}/trained_b3.csv".format(config.CONFIG_ID), b3_val, delimiter=",")
    np.savetxt("./model/{0}/trained_W4.csv".format(config.CONFIG_ID), np.ravel(W4_val), delimiter=",")
    np.savetxt("./model/{0}/trained_b4.csv".format(config.CONFIG_ID), b4_val, delimiter=",")
    np.savetxt("./model/{0}/trained_Wfc1.csv".format(config.CONFIG_ID), np.ravel(W_fc1_val), delimiter=",")
    np.savetxt("./model/{0}/trained_bfc1.csv".format(config.CONFIG_ID), b_fc1_val, delimiter=",")
    np.savetxt("./model/{0}/trained_Wfc2.csv".format(config.CONFIG_ID), np.ravel(W_fc2_val), delimiter=",")
    np.savetxt("./model/{0}/trained_bfc2.csv".format(config.CONFIG_ID), b_fc2_val, delimiter=",")


# def recordExperimentalSetup():
    
#     msgLst = []
#     msgLst.append('Configurations:')
#     msgLst.append('\tConfiguration ID = {0}'.format(config.CONFIG_ID))
#     msgLst.append('\tpreproc:')
#     msgLst.append('\t\tMIN_LONG = {0}'.format(config.MIN_LONG))
#     msgLst.append('\t\tMAX_LONG = {0}'.format(config.MAX_LONG))
#     msgLst.append('\t\tMIN_LAT = {0}'.format(config.MIN_LAT))
#     msgLst.append('\t\tMAX_LAT = {0}'.format(config.MAX_LAT))
#     msgLst.append('\t\tGRAY_SCALE = {0}'.format(config.GRAY_SCALE))
#     msgLst.append('\t\tSTEP = {0}'.format(config.STEP))
#     msgLst.append('\t\tTIME_WINDOW = {0}'.format(config.TIME_WINDOW))
#     msgLst.append('\t\tRANDOM_SEED = {0}.'.format(config.SEED))
#     msgLst.append('\tconv:')
#     msgLst.append('\t\tBATCH_SIZE = {0}'.format(config.BATCH_SIZE))
#     msgLst.append('\t\tCONV1_FILTER_WIDTH = {0}'.format(config.CONV1_FILTER_WIDTH))
#     msgLst.append('\t\tCONV1_FILTER_HEIGHT = {0}'.format(config.CONV1_FILTER_HEIGHT))
#     msgLst.append('\t\tCONV1_OUT_CH_NUM = {0}'.format(config.CONV1_OUT_CH_NUM))
#     msgLst.append('\t\tCONV2_FILTER_WIDTH = {0}'.format(config.CONV2_FILTER_WIDTH))
#     msgLst.append('\t\tCONV2_FILTER_HEIGHT = {0}'.format(config.CONV2_FILTER_HEIGHT))
#     msgLst.append('\t\tCONV2_OUT_CH_NUM = {0}'.format(config.CONV2_OUT_CH_NUM))
#     msgLst.append('\t\tCONV3_FILTER_WIDTH = {0}'.format(config.CONV3_FILTER_WIDTH))
#     msgLst.append('\t\tCONV3_FILTER_HEIGHT = {0}'.format(config.CONV3_FILTER_HEIGHT))
#     msgLst.append('\t\tCONV3_OUT_CH_NUM = {0}'.format(config.CONV3_OUT_CH_NUM))
#     msgLst.append('\t\tCONV4_FILTER_WIDTH = {0}'.format(config.CONV4_FILTER_WIDTH))
#     msgLst.append('\t\tCONV4_FILTER_HEIGHT = {0}'.format(config.CONV4_FILTER_HEIGHT))
#     msgLst.append('\t\tCONV4_OUT_CH_NUM = {0}'.format(config.CONV4_OUT_CH_NUM))
#     msgLst.append('\t\tFC_FILTER_WIDTH = {0}'.format(config.FC_FILTER_WIDTH))
#     msgLst.append('\t\tFC_FILTER_HEIGHT = {0}'.format(config.FC_FILTER_HEIGHT))
#     msgLst.append('\t\tFULLY_CONNECTED_NUM = {0}'.format(config.FULLY_CONNECTED_NUM))
#     msgLst.append('\t\tDROP_OUT_PROB = {0}'.format(config.DROP_OUT_PROB))
#     msgLst.append('\tTRANING_SET_RATE = {0}'.format(config.TRANING_SET_RATE))
#     msgLst.append('\tITERATION = {0}'.format(config.ITERATION))

#     print('\nData Information:')
#     print('\tTOTAL SET SIZE = {0}'.format(len(rgb_x)))
#     print('\t\tTOTAL TRUE/FALSE COUNT = {0}/{1}'.format(total_y_true, total_y_false))
#     print('\tTRAINING SET SIZE = {0}'.format(len(rgb_x_train)))
#     print('\tTEST SET SIZE = {0}'.format(len(rgb_x_test)))
#     print('\t\tTOTAL TRUE/FALSE COUNT IN TEST SET = {0}/{1}'.format(len(x_t), len(x_f)))

#     msgLst.append('\nData Information:')
#     msgLst.append('\tTOTAL SET SIZE = {0}'.format(len(rgb_x)))
#     msgLst.append('\t\tTOTAL TRUE/FALSE COUNT = {0}/{1}'.format(total_y_true, total_y_false))
#     msgLst.append('\tTRAINING SET SIZE = {0}'.format(len(rgb_x_train)))
#     msgLst.append('\tTEST SET SIZE = {0}'.format(len(rgb_x_test)))
#     msgLst.append('\t\tTOTAL TRUE/FALSE COUNT IN TEST SET = {0}/{1}'.format(len(x_t), len(x_f)))

#     return msgLst

def plotLayer(sess, f, x_sel):
    
    val = sess.run(f, feed_dict={x: x_sel})    
    
    num_filters = val.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    
    fig, axes = plt.subplots(num_grids, num_grids)

    for i, ax in enumerate(axes.flat):
        if i<num_filters:
            img = val[0, :, :, i]
            ax.imshow(img, interpolation='nearest', cmap='binary')
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle(str(f), fontsize=8)
    plt.show()


# def recordExperimentalResult(sess, msgLst):
#     print('\nTraining complete!\n')
#     #f_train_acc = eval_acc(sess, accuracy, rgb_x_train, rgb_y_train)
#     f_train_acc, _p, _r = eval_metrics(sess, rgb_x_train, rgb_y_train)
#     print('Training accuracy = {0}'.format(f_train_acc))
#     msgLst.append('Training accuracy = {0}'.format(f_train_acc))

#     #f_test_acc = eval_acc(sess, accuracy, rgb_x_test, rgb_y_test)
#     #f_true_only_acc = eval_acc(sess, accuracy, x_t, y_t)
#     #f_false_only_acc = eval_acc(sess, accuracy, x_f, y_f)
    
#     pred = sess.run(prediction, feed_dict={x:rgb_x_test, keep_prob:1.0})
#     labels = tf.argmax(rgb_y_test,1)

#     f_test_acc, f_test_precision, f_test_recall = eval_metrics(sess, rgb_x_test, rgb_y_test)

#     print('\n-------------------------------------------------')
#     print('Final conv: test accuracy: %g'%f_test_acc)
#     print('Final conv: test precision: %g'%f_test_precision)
#     print('Final conv: test recall: %g'%f_test_recall)
#     print('Total spent time: {0} (sec)'.format(time.time()-start_time))
#     msgLst.append('\n-------------------------------------------------')
#     msgLst.append('Final conv: test accuracy: %g'%f_test_acc)
#     msgLst.append('Final conv: test precision: %g'%f_test_precision)
#     msgLst.append('Final conv: test recall: %g'%f_test_recall)
#     msgLst.append('Total spent time: {0} (sec)'.format(time.time()-start_time))

#     return msgLst


def nearestneighborDrawing():
    sess = tf.Session()

    np.random.seed(13)  # set seed for reproducibility
    train_size = 1000
    test_size = 102
    rand_train_indices = np.random.choice(len(rgb_x_train), train_size, replace=False)
    print(rand_train_indices)
    rand_test_indices = np.random.choice(len(rgb_x_test), test_size, replace=False)
    
    x_vals_train = []
    x_vals_test = []
    y_vals_train =[]
    y_vals_test = []

    for t in rand_train_indices:
        x_vals_train.append(rgb_x_train[t])
        y_vals_train.append(rgb_y_train[t])

    for t in rand_test_indices:
        x_vals_test.append(rgb_x_test[t])
        y_vals_test.append(rgb_y_test[t])

    x_data_train = tf.placeholder(shape=[None, config.STEP*config.STEP*3], dtype=tf.float32)
    x_data_test = tf.placeholder(shape=[None, config.STEP*config.STEP*3], dtype=tf.float32)
    y_target_train = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    y_target_test = tf.placeholder(shape=[None, 2], dtype=tf.float32)

    distance = tf.reduce_sum(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1))), axis=2)

    # Get min distance index (Nearest neighbor)
    top_k_xvals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=10)
    prediction_indices = tf.gather(y_target_train, top_k_indices)

    # Predict the mode category
    count_of_predictions = tf.reduce_sum(prediction_indices, axis=1)
    prediction = tf.argmax(count_of_predictions, axis=1)

    # Calculate how many loops over training data
    num_loops = int(np.ceil(len(x_vals_test)/6))

    test_output = []
    actual_vals = []
    for i in range(num_loops):
        min_index = i*6
        max_index = min((i+1)*6,len(x_vals_train))
        x_batch = x_vals_test[min_index:max_index]
        y_batch = y_vals_test[min_index:max_index]
        predictions = sess.run(prediction, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch,
                                            y_target_train: y_vals_train, y_target_test: y_batch})
        test_output.extend(predictions)
        actual_vals.extend(np.argmax(y_batch, axis=1))

    accuracy = sum([1./test_size for i in range(test_size) if test_output[i]==actual_vals[i]])
    print('Accuracy on test set: ' + str(accuracy))

    actuals = np.argmax(y_batch, axis=1)

    Nrows = 2
    Ncols = 3
    for i in range(len(actuals)):
        plt.subplot(Nrows, Ncols, i+1)
        g = np.divide(x_batch[i], 255.)
        plt.imshow(np.reshape(g, [32,32,3]), cmap='Greys_r')
        plt.title('Actual: ' + str(actuals[i]) + ' Pred: ' + str(predictions[i]),
                                fontsize=10)
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
    
    plt.show()



#rgb_x_train, rgb_y_train, rgb_x_test, rgb_y_test = get_samples()
rgb_x_train, rgb_y_train, rgb_x_test, rgb_y_test, rgb_ids = get_samples()

#nearestneighborDrawing()

# msgLst = recordExperimentalSetup()

it = len(rgb_x_train)//(config.BATCH_SIZE)

x = tf.placeholder(tf.float32, shape=[None, config.STEP*config.STEP*3])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

def shift(rgb_x, rgb_y):
    
    rgb_x_firstBat = rgb_x[:config.BATCH_SIZE]
    rgb_y_firstBat = rgb_y[:config.BATCH_SIZE]

    rgb_x = rgb_x[config.BATCH_SIZE:] + rgb_x_firstBat
    rgb_y = rgb_y[config.BATCH_SIZE:] + rgb_y_firstBat 

    return rgb_x, rgb_y

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # small noise for symmetry breaking
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape) # prevent dead neuron 
    return tf.Variable(initial)

def conv2d(x,W,s):
    return tf.nn.conv2d(x,W, strides=[1,s,s,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def eval_acc(sess, acc_f, x_lst, y_lst):
    total_len = len(x_lst)
    num_batch = total_len//10000 + 1
    last_batch_size = total_len - 10000 * (num_batch-1) 
    total_predic = 0
    for i in range(num_batch):
        k = 10000*i
        j = min(k+10000, total_len)
        f_partial_acc = sess.run(acc_f, feed_dict={x:x_lst[k:j], y_:y_lst[k:j], keep_prob:1.0})
        total_predic += (j-k) * f_partial_acc
    
    return total_predic/total_len

def beautyCM(cm, ind=['True pos', 'True neg'], cols=['Pred pos', 'Pred neg']):
    return pd.DataFrame(cm, index=ind, columns=cols)

def true_positive(cm):
    return cm.get_value(index='True pos', col='Pred pos')

def true_negative(cm):
    return cm.get_value(index='True neg', col='Pred neg')

def false_positive(cm):
    return cm.get_value(index='True neg',col='Pred pos')

def false_negative(cm):
    return cm.get_value(index='True pos', col='Pred neg')


def eval_metrics(sess, x_lst, y_lst):
    total_len = len(x_lst)
    num_batch = total_len//10000 + 1
    last_batch_size = total_len - 10000 * (num_batch-1) 
    
    true_positive = 0.0
    false_positive = 0.0
    false_negative = 0.0
    true_negative = 0.0

    for i in range(num_batch):
        k = 10000*i
        j = min(k+10000, total_len)

        cm = confusion_matrix_tf.eval(feed_dict={x:x_lst[k:j], y_:y_lst[k:j], keep_prob:1.0})
        #pred = sess.run(prediction, feed_dict={x:x_lst[k:j], keep_prob:1.0})
        #label = y_lst[k:j]

        

        #tp_tensor = tf.metrics.true_positives(labels=tf.argmax(label,1) , predictions=tf.argmax(pred,1))
        #fp_tensor = tf.metrics.false_positives(labels=tf.argmax(label,1) , predictions=tf.argmax(pred,1))
        #fn_tensor = tf.metrics.false_negatives(labels=tf.argmax(label,1) , predictions=tf.argmax(pred,1))
        #tn_tensor = tf.metrics.true_negatives_at_thresholds(labels=tf.argmax(label,1) , predictions=tf.argmax(pred,1), thresholds= [0 for i in range(j-k)])
        #tf.global_variables_initializer().run(session=sess)
        #tf.local_variables_initializer().run(session=sess)


        #tp = sess.run(tp_tensor)
        #fp = sess.run(fp_tensor)
        #fn = sess.run(fn_tensor)
        #tn = sess.run(tn_tensor)

        #print(tp, fp, fn)
        #print(tp[0], fp[0], fn[0])
        #tn = (j-k) - tp[0] - fp[0] - fn[0]


        cm_run = beautyCM(cm)

        #print(cm_run)

        #a = true_positive(cm_run)
        #b = false_positive(cm_run)
        #c = false_negative(cm_run)
        #d = true_negative(cm_run)

        a = cm_run.iat[0,0]
        b = cm_run.iat[0,1]
        c = cm_run.iat[1,0]
        d = cm_run.iat[1,1]

        #print(a,b,c,d)

        true_positive += a
        false_positive += b
        false_negative += c
        true_negative += d
    
    acc =  (true_positive + true_negative)/(total_len)
    prec = (true_positive )/(true_positive+false_positive)
    rec = (true_positive )/(true_positive+false_negative)
    
    return acc, prec, rec

with tf.device('/gpu:0'):
    W_conv1 = weight_variable([config.CONV1_FILTER_WIDTH,config.CONV1_FILTER_HEIGHT,3,config.CONV1_OUT_CH_NUM])
    b_conv1 = bias_variable([config.CONV1_OUT_CH_NUM])

    x_image = tf.reshape(x, [-1,config.STEP,config.STEP,3])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([config.CONV2_FILTER_WIDTH,config.CONV2_FILTER_HEIGHT,config.CONV1_OUT_CH_NUM,config.CONV2_OUT_CH_NUM])
    b_conv2 = bias_variable([config.CONV2_OUT_CH_NUM])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 1) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_conv3 = weight_variable([config.CONV3_FILTER_WIDTH,config.CONV3_FILTER_HEIGHT,config.CONV2_OUT_CH_NUM,config.CONV3_OUT_CH_NUM])
    b_conv3 = bias_variable([config.CONV3_OUT_CH_NUM])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, 1) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    W_conv4 = weight_variable([config.CONV4_FILTER_WIDTH,config.CONV4_FILTER_HEIGHT,config.CONV3_OUT_CH_NUM, config.CONV4_OUT_CH_NUM])
    b_conv4 = bias_variable([config.CONV4_OUT_CH_NUM])
    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4, 1) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4)

    W_fc1 = weight_variable([config.FC_FILTER_WIDTH*config.FC_FILTER_HEIGHT*config.CONV4_OUT_CH_NUM, config.FULLY_CONNECTED_NUM])
    b_fc1 = bias_variable([config.FULLY_CONNECTED_NUM])

    h_pool4_flat = tf.reshape(h_pool4, [-1, config.FC_FILTER_WIDTH*config.FC_FILTER_HEIGHT*config.CONV4_OUT_CH_NUM])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1)+b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([config.FULLY_CONNECTED_NUM,2])
    b_fc2 = bias_variable([2])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    #y_conv = tf.nn.sigmoid(y_conv)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    )

    #loss = tf.reduce_sum(y_*tf.log(y_conv) + (1-y_)*tf.log(y_conv))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
log_topology()
with tf.Session() as sess:    
    sess.run(tf.global_variables_initializer())
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    confusion_matrix_tf = tf.confusion_matrix(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #prediction = y_conv
    prediction = y_conv
    #labels = tf.argmax(y_,1)
    print('\n-------------------------------------------------')
    # msgLst.append('\n-------------------------------------------------')
    start_time = time.time()
    for j in range(config.ITERATION):
        print('Iteration ({0}/{1}) starts training...'.format((j+1), config.ITERATION))
        # msgLst.append('Iteration ({0}/{1}) :'.format((j+1), config.ITERATION))
        for i in range(it):
            xx = rgb_x_train[config.BATCH_SIZE*i:config.BATCH_SIZE*i+config.BATCH_SIZE]
            yy = rgb_y_train[config.BATCH_SIZE*i:config.BATCH_SIZE*i+config.BATCH_SIZE]
            train_step.run(feed_dict={x: xx, y_: yy, keep_prob:config.DROP_OUT_PROB})

        if j % 5 == 0 :
            #p_train = sess.run(prediction, feed_dict={x:rgb_x_train, keep_prob:1.0})
            #p_label = tf.argmax(rgb_y_train,1)
            f_train_acc, _p, _r = eval_metrics(sess, rgb_x_train, rgb_y_train)
            #f_train_acc = eval_acc(sess, accuracy, rgb_x_train, rgb_y_train)
            print('\tIter {0} training accuracy = {1}'.format((j+1), f_train_acc))
            print('\tIter {0} training precision = {1}'.format((j+1), _p))
            print('\tIter {0} training recall = {1}'.format((j+1), _r))

            #print(h_conv1.eval(feed_dict={x: rgb_x_train[:1]}))

        rgb_x_train, rgb_y_train = shift(rgb_x_train, rgb_y_train)

    print('TRAINING IS COMPLETE!')

    it2 = len(rgb_x_test)//5

    for i in range(it2):
        r = []
        r.append(rgb_ids[5*i])
        
        x1, x2, x3, x4, x5 = rgb_x_test[5*i:5*i+5]
        y1, y2, y3, y4, y5 = rgb_y_test[5*i:5*i+5]

        xs = [x1, x2, x3, x4, x5]
        ys = [y1, y2, y3, y4, y5]

        lab = -1
        for j, val in enumerate(ys):
            if val == [1,0]:
                lab = j
                break
        
        r.append([lab])
        
        pred = y_conv
        pred = pred.eval(feed_dict = {x: xs, keep_prob:1.0})
        
        pred = pred.tolist()

        y_predicted = []
        for p in pred:
            if p == 1: # [0, 1]
                y_predicted.append(0)
            elif p== 0 : # [1, 0]
                y_predicted.append(1)
            else:
                y_predicted.append(p)
        r = r+y_predicted

        print(r)



    #f_train_acc = sess.run(accuracy, feed_dict={x:rgb_x_train, y_:rgb_y_train, keep_prob:1.0})
    # msgLst = recordExperimentalResult(sess, msgLst)

    ANUM=155
    plotLayer(sess, x_image, rgb_x_train[ANUM:ANUM+1])

    plotLayer(sess, h_conv1, rgb_x_train[ANUM:ANUM+1])
    plotLayer(sess, h_pool1, rgb_x_train[ANUM:ANUM+1])

    plotLayer(sess, h_conv2, rgb_x_train[ANUM:ANUM+1])
    plotLayer(sess, h_pool2, rgb_x_train[ANUM:ANUM+1])
    
    plotLayer(sess, h_conv3, rgb_x_train[ANUM:ANUM+1])
    plotLayer(sess, h_pool3, rgb_x_train[ANUM:ANUM+1])

    plotLayer(sess, h_conv4, rgb_x_train[ANUM:ANUM+1])
    plotLayer(sess, h_pool4, rgb_x_train[ANUM:ANUM+1])

    print(rgb_y_train[ANUM:ANUM+1])

    z = np.asarray(rgb_x_train[ANUM], dtype='uint8')
    sz = np.asarray(z.reshape(128,128,3))
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    plt.title('Original Image')
    plt.imshow(sz)
    plt.show()
    '''

    # log_param(sess)
    
    # file_name_suffix = time.strftime('%Y-%m-%d_%H-%M-%S')
    # log_file = open('./result/log_{0}_{1}.txt'.format(config.CONFIG_ID, file_name_suffix), 'w')
    # for l in msgLst:
    #     log_file.write(l+'\n')

    # log_file.close()
'''

