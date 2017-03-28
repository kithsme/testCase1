import csv
import numpy as np
import random
from PIL import Image
from datetime import datetime as dt
from datetime import timedelta as td

orderdata = []
#f = open('C:/Users/kiths/Documents/논문/데이터/ORDER_DATA.csv')
#f = open('C:/Users/kiths/Desktop/ridertest.csv')
f = open('/users/tkim/Downloads/order_data.csv')
csvReader = csv.reader(f)

for row in csvReader:
    orderdata.append(row)

f.close

MIN_LONG = 127.02
MAX_LONG = 127.141

MIN_LAT = 37.485
MAX_LAT = 37.54

STEP = 32
TIME_WINDOW = 15

SEPATED_MODE='w' # 's'

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
        rgbArray = np.zeros((STEP, STEP, 3),'uint8')
        from_long = float(aRow[2])
        from_lat = float(aRow[3])
        from_i, from_j = map(from_long, from_lat)
        to_long = float(aRow[5])
        to_lat = float(aRow[6])
        to_i, to_j = map(to_long, to_lat)
        if from_i==to_i and from_j==to_j:
            rgbArray[from_i][from_j][0] = 255
            rgbArray[to_i][to_j][2] = 255
        else:
            rgbArray[from_i][from_j][0] = 255
            #rgbArray[from_i][from_j][1] = 0
            #rgbArray[from_i][from_j][2] = 0
            #rgbArray[to_i][to_j][0] = 0
            #rgbArray[to_i][to_j][1] = 0
            rgbArray[to_i][to_j][2] = 255

        rgbDic[aRow[1]] = sum(sum(rgbArray.tolist(),[]),[])
        coorDic[aRow[1]] = [from_long, from_lat, to_long, to_lat]
        if not aRow[7] in adjDic:
            adjDic[aRow[7]] = [(aRow[1], parse_date(aRow[0]), 
            parse_date(aRow[8]), parse_date(aRow[9]), parse_date(aRow[10]))]
        else:
            adjDic[aRow[7]].append((aRow[1], parse_date(aRow[0]), 
            parse_date(aRow[8]), parse_date(aRow[9]), parse_date(aRow[10])))
        #im = Image.fromarray(rgbArray)
        #im.save("C:/Users/kiths/Documents/visualstudiocode-tensorflow/my_file_{0}.png".format(aRow[0]))
    weak, strong = binned(adjDic)

    sep = separated(matrix, weak, strong, mode=SEPATED_MODE, tw=TIME_WINDOW)

    return rgbDic, coorDic, sep, weak, strong

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
    
    random.shuffle(ret)
    return ret

rgb_dic, coord_dic, sep, weak, strong = makeInputs(orderdata)

'''
print(sep)
print('---------------------------------------')
print(weak)
print('---------------------------------------')
print(strong)
print('---------------------------------------')
'''
print( len(sep), len(weak), len(strong) )

input1 = pair_to_xy(coord_dic, sep, weak, strong, mode=SEPATED_MODE)
input2 = pair_to_xy(rgb_dic, sep, weak, strong, mode=SEPATED_MODE)


'''
with open('C:/Users/kiths/Desktop/book2_proc_rgb.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for a in rgb_dic.items():
        writer.writerow([a[0]]+a[1])

with open('C:/Users/kiths/Desktop/book2_proc_coord.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for a in coord_dic.items():
        writer.writerow([a[0]]+a[1])
'''
