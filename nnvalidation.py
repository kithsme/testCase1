import sklearn.neural_network
import preproc
import numpy as np
import random
import conf


coord_xy = preproc.preproc()
coord_xy = np.asarray(coord_xy)

coord_xy = coord_xy[:, :-2]
coord_xy = coord_xy.astype(np.float)
x = []
y = []

zero = [0,1,2,3,4,5,6,7]
one = [0,1,4,5,2,3,6,7]
two = [4,5,0,1,2,3,6,7]
three = [0,1,4,5,6,7,2,3]
four = [4,5,0,1,6,7,2,3]

for row in coord_xy:
    if row[-1] == 0.0:
        #x.append(np.append(row,[0]))
        #x.append(np.append(row[:-1],[1,0]))
        #x.append(np.append(row[:-1],[2,0]))
        #x.append(np.append(row[:-1],[3,0]))
        #x.append(np.append(row[:-1],[4,0]))
        tmp = [row[i] for i in zero]
        tmp.append(1)
        x.append(tmp)
        tmp = [row[i] for i in one]
        tmp.append(0)
        x.append(tmp)

    elif row[-1] == 1.0:
        tmp = [row[i] for i in one]
        tmp.append(1)
        x.append(tmp)
        tmp = [row[i] for i in zero]
        tmp.append(0)
        x.append(tmp)
    elif row[-1] == 2.0:
        tmp = [row[i] for i in two]
        tmp.append(1)
        x.append(tmp)
        tmp = [row[i] for i in zero]
        tmp.append(0)
        x.append(tmp)
    elif row[-1] == 3.0:
        tmp = [row[i] for i in three]
        tmp.append(1)
        x.append(tmp)
        tmp = [row[i] for i in zero]
        tmp.append(0)
        x.append(tmp)
    elif row[-1] == 4.0:
        tmp = [row[i] for i in four]
        tmp.append(1)
        x.append(tmp)
        tmp = [row[i] for i in zero]
        tmp.append(0)
        x.append(tmp)

x = np.asarray(x)

print(x)

random.seed(1120)
random.shuffle(x)

bar = int(len(x)*0.7)

x_train = x[:bar,:-1]
y_train = x[:bar,-1]
y_train = y_train.astype(np.int)

x_test = x[bar:, :-1]
y_test = x[bar:,-1]
y_test = y_test.astype(np.int)

nn = sklearn.neural_network.MLPClassifier()
nn.fit(x_train,y_train)

y_out = nn.predict(x_test)

a = 0
b = 0
c = 0
d = 0
t = 0

for i, y_ in enumerate(y_out):
    if y_ == y_test[i] and y_ == 1:
        a +=1
    elif y_ == y_test[i] and y_ == 0:
        d += 1
    elif y_ != y_test[i] and y_ == 1:
        b += 1
    elif y_ != y_test[i] and y_ == 0:
        c += 1
    t+= 1

print(a,b,c,d,t)

print((a+d)/t)
print(0 if (a+b)==0 else (a)/(a+b))
print(0 if (a+c)==0 else (a)/(a+c))
