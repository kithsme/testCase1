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
