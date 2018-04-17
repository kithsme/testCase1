import numpy as np

import random

def seq(coord_xy, norm=False):
    zero = [0,1,2,3,4,5,6,7]
    one = [0,1,4,5,2,3,6,7]
    two = [4,5,0,1,2,3,6,7]
    three = [0,1,4,5,6,7,2,3]
    four = [4,5,0,1,6,7,2,3]

    x=[]
    y=[]

    for row in coord_xy:
        if row[-1] == 0.0:
            #x.append(np.append(row,[0]))
            #x.append(np.append(row[:-1],[1,0]))
            #x.append(np.append(row[:-1],[2,0]))
            #x.append(np.append(row[:-1],[3,0]))
            #x.append(np.append(row[:-1],[4,0]))
            tmp = [row[i] for i in zero]
            #tmp.append(1.0)
            tmp.append(1)
            x.append(tmp)
            tmp = [row[i] for i in one]
            #tmp.append(2.0)
            tmp.append(0)
            x.append(tmp)
        elif row[-1] == 1.0:
            tmp = [row[i] for i in zero]
            #tmp.append(1.0)
            tmp.append(0)
            x.append(tmp)
            tmp = [row[i] for i in one]
            #tmp.append(2.0)
            tmp.append(1)
            x.append(tmp)
        elif row[-1] == 2.0:
            tmp = [row[i] for i in zero]
            #tmp.append(1.0)
            tmp.append(0)
            x.append(tmp)
            tmp = [row[i] for i in two]
            #tmp.append(3.0)
            tmp.append(1)
            x.append(tmp)
        elif row[-1] == 3.0:
            tmp = [row[i] for i in zero]
            #tmp.append(1.0)
            tmp.append(0)
            x.append(tmp)
            tmp = [row[i] for i in three]
            #tmp.append(4.0)
            tmp.append(1)
            x.append(tmp)
        elif row[-1] == 4.0:
            tmp = [row[i] for i in zero]
            #tmp.append(1.0)
            tmp.append(0)
            x.append(tmp)
            tmp = [row[i] for i in four]
            #tmp.append(5.0)
            tmp.append(1)
            x.append(tmp)
    
    
    x = np.asarray(x)

    return x, y

def proc_coord(train_coord_xy, test_coord_xy, norm=False):

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

    return x_train, x_test, y_train, y_test