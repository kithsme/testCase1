import sklearn.neural_network
import preproc
import numpy as np
import random
import conf
import os
import genImage
import conf
import time


config = conf.conf()

coord_xy = preproc.preproc()
if not os.path.exists("./model"):
    os.makedirs("./model")
if not os.path.exists("./model/{0}".format(config.CONFIG_ID)):
    os.makedirs("./model/{0}".format(config.CONFIG_ID))
log_file = open('./model/{0}/preproc_config.csv'.format(config.CONFIG_ID), 'w')
content = [str(config.MIN_LONG), str(config.MAX_LONG), str(config.MIN_LAT), str(config.MAX_LAT), str(config.GRAY_SCALE), str(config.STEP)]
log_file.write(','.join(content))
log_file.close()
con = genImage.getConfig(config.CONFIG_ID)
gm = genImage.draw_map(coord_xy,con)
rgb_x, rgb_y = genImage.xy_to_rgbArray(coord_xy, gm, con)

randidx = [i for i in range(len(rgb_x))]
bar = int(len(rgb_x)*0.7)

rgb_x = np.asarray(rgb_x)
rgb_x = rgb_x.astype(np.float)

rgb_y = np.asarray(rgb_y)
rgb_y = rgb_y.astype(np.int)

train_x = []
train_y = []
test_x = []
test_y = []

for i in randidx[:bar]:
    train_x.append(rgb_x[i])
    if rgb_y[i,0] == 1:
        train_y.append(1)
    else:
        train_y.append(0)

for i in randidx[bar:]:
    test_x.append(rgb_x[i])
    if rgb_y[i,0] == 1:
        test_y.append(1)
    else:
        test_y.append(0)

train_x = np.asarray(train_x)
test_x = np.asarray(test_x)
train_y = np.asarray(train_y)
test_y = np.asarray(test_y)

train_y = train_y.astype(np.int)
test_y = test_y.astype(np.int)


nn = sklearn.neural_network.MLPClassifier()
start_time = time.time()

nn.fit(train_x,train_y)
end_time = time.time()
print(end_time-start_time)
y_out = nn.predict(test_x)

a = 0
b = 0
c = 0
d = 0
t = 0

for i, y_ in enumerate(y_out):
    if y_ == test_y[i] and y_ == 1:
        a +=1
    elif y_ == test_y[i] and y_ == 0:
        d += 1
    elif y_ != test_y[i] and y_ == 1:
        b += 1
    elif y_ != test_y[i] and y_ == 0:
        c += 1
    t+= 1

print(a,b,c,d,t)

print((a+d)/t)
print(0 if (a+b)==0 else (a)/(a+b))
print(0 if (a+c)==0 else (a)/(a+c))