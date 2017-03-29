import csv
import os
import numpy as np
import random
from PIL import Image
from datetime import datetime as dt
from datetime import timedelta as td

MIN_LONG = 127.02
MAX_LONG = 127.141
MIN_LAT = 37.485
MAX_LAT = 37.54
STEP = 32
TIME_WINDOW = 15
SEPATED_MODE='s' # 'w'

def makeInputs(matrix):
    
    rgbDic={}
    coorDic={}
    adjDic = {}

    for aRow in matrix:
        """
        aRow[0]: order creation timestamp
        aRow[1]: order id 
        aRow[2,3]: pickup long,lat
        aRow[4]: order noticed timestamp
        aRow[5,6]: destination long, lat
        aRow[7]: deliverer id 
        aRow[8,9,10]: catched timestamp, pickedup timestamp, delivered timestamp
        """
        from_long = float(aRow[2])
        from_lat = float(aRow[3])
        to_long = float(aRow[5])
        to_lat = float(aRow[6])

        coorDic[aRow[1]] = [from_long, from_lat, to_long, to_lat]
        if not aRow[7] in adjDic:
            adjDic[aRow[7]] = [(aRow[1], parse_date(aRow[0]), 
            parse_date(aRow[8]), parse_date(aRow[9]), parse_date(aRow[10]))]
        else:
            adjDic[aRow[7]].append((aRow[1], parse_date(aRow[0]), 
            parse_date(aRow[8]), parse_date(aRow[9]), parse_date(aRow[10])))
        
    weak, strong = binned(adjDic)

    sep = separated(matrix, weak, strong, mode=SEPATED_MODE, tw=TIME_WINDOW)

    return coorDic, sep, weak, strong

def parse_date(datetimeStr):
    if type(datetimeStr) is type(dt):
        return datetimeStr
    for fmt in ('%Y-%m-%d %H:%M', '%Y.%m.%d %H:%M' ,'%H:%M:%S'):
        try:
            return dt.strptime(datetimeStr, fmt)
        except ValueError:
            pass
    raise ValueError('No valide date format found for %s'%(datetimeStr))

def separated(matrix, weak, strong, mode='w', tw=15):
    sepa = []
    twindow = td(seconds=tw*60)
    for i in range(len(matrix)-1):
        aRow = matrix[i]
        aId = aRow[1]
        sTa = parse_date(aRow[8]) # catched timestamp of previous order
        for j in range(i+1, len(matrix)):
            bRow = matrix[j]
            bId = bRow[1]
            sTb = parse_date(bRow[8])
            if sTa < sTb < sTa+twindow :
                cand = [aId, bId]
                if mode == 'w' and cand not in weak and cand not in strong:
                    sepa.append(cand)
                elif mode == 's' and cand not in strong:
                    sepa.append(cand)
            else:
                break
    return sepa

def binned(adjDic):
    """
    a[0]: order id
    a[1]: creation timestamp
    a[2,3,4]: catched timestamp, pickedup timestamp, delivered timestamp
    """
    weak=[]
    strong=[]
    prev=None
    for key in adjDic:
        prev = adjDic[key][0]
        for a in adjDic[key]:
            if prev[2]<a[2] and a[2] < prev[3]:
                strong.append( [prev[0], a[0]] )
            elif prev[2]<a[2] and a[2] < prev[4]:
                weak.append( [prev[0], a[0]] )
            prev = a

    return weak, strong

def map(lon, lat):
    diff_long = (MAX_LONG-MIN_LONG)/(STEP-1)
    diff_lat = (MAX_LAT-MIN_LAT)/(STEP-1)

    a = int(round((lon - MIN_LONG)/diff_long))
    b = int(round((lat - MIN_LAT)/diff_lat))

    if a>=STEP or b>=STEP:
        print(lon,lat)
    return a,b

def xy_to_rgbArray(xy):
    """
    return tuples!!!
    """
    ret=[]
    count = 1
    for a in xy:
        """
        a[0,1]: previous from long,lat
        a[2,3]: previous to long,lat
        a[4,5]: following from long,lat
        a[6,7]: following to long,lat
        a[8]: y (1 or 0)  
        """
        rgb_array = np.zeros((STEP,STEP,3), 'uint8')
        p_from_i, p_from_j = map(a[0], a[1])
        p_to_i, p_to_j = map(a[2], a[3])
        q_from_i, q_from_j = map(a[4], a[5])
        q_to_i, q_to_j = map(a[6], a[7])

        # if two orders share a point, previous one always overwrite it
        rgb_array[q_from_i][q_from_j][0] = 255
        rgb_array[q_to_i][q_to_j][2] = 255

        rgb_array[p_from_i][p_from_j][0] = 255
        rgb_array[p_from_i][p_from_j][1] = 255
        rgb_array[p_to_i][p_to_j][1] = 255
        rgb_array[p_to_i][p_to_j][2] = 255
        ret.append( (rgb_array, a[8]) )
        count +=1

        # print for debugging and illustrative examples
        if count % 739 == 0:
            im = Image.fromarray(rgb_array)
            im.save("/result/fig/test1_fig/{0}_fig_{1}.png".format(a[8], count))

    return ret

def pair_to_xy(dic, sep, weak, strong, mode='w'):

    ret = []
    for pair in sep:
        a, b = pair[0], pair[1]
        y=[0] # bc it comes from sep, y = 0
        x = dic[a] + dic[b] + y
        ret.append(x)
    
    for pair in strong:
        a, b = pair[0], pair[1]
        y=[1] # bc it comes from strong binned, y = 1
        x = dic[a] + dic[b] + y
        ret.append(x)

    if mode=='w':
        for pair in weak:
            a, b = pair[0], pair[1]
            y=[1] # bc it comes from weak binned, y = 1
            x = dic[a] + dic[b] + y
            ret.append(x)
    else:
        for pair in weak:
            a, b = pair[0], pair[1]
            y=[0] # bc it comes from weak binned, but policy is strong so y = 0
            x = dic[a] + dic[b] + y
            ret.append(x)
    
    random.shuffle(ret)
    return ret

def write_csv(lst, rgb=0):
    random.shuffle(lst)
    slicingInd = int(len(lst) * 0.75)
    ltype = 'coords' if rgb==0 else 'rgb'
    filename = './result/{0}_training_data.csv'.format(ltype)
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for a in lst[:slicingInd]:
            writer.writerow(a)

    filename = './result/{0}_test_data.csv'.format(ltype)
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for a in lst[slicingInd:]:
            writer.writerow(a)

def preproc(rgb=0):
    orderdata = []
    f = open('./data/order_data.csv')
    csvReader = csv.reader(f)

    for row in csvReader:
        orderdata.append(row)
    f.close

    coord_dic, sep, weak, strong = makeInputs(orderdata)
    # print( len(sep), len(weak), len(strong) )  #check for values

    coord_xy = pair_to_xy(coord_dic, sep, weak, strong, mode=SEPATED_MODE)
    if rgb==1:
        rgb_tuples = xy_to_rgbArray(coord_xy)
        print('preprocessing complete w/ coordinates & rgb inputs')
        #return coord_xy, rgb_tuples
    else:
        print('preprocessing complete w/ coordinates inputs')
        write_csv(coord_xy, rgb)
