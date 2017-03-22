import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime as dt

matrix = []
#f = open('C:/Users/kiths/Documents/논문/데이터/ORDER_DATA.csv')
f = open('C:/Users/kiths/Desktop/ridertest.csv')
csvReader = csv.reader(f)

for row in csvReader:
    matrix.append(row)

f.close
'''
git test
'''

MIN_LONG = 127.02
MAX_LONG = 127.141

MIN_LAT = 37.485
MAX_LAT = 37.54

STEP = 32

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
        format_creationdatetime = '%Y-%m-%d %H:%M'
        format_timestamp = '%H:%M:%S'
        if not aRow[7] in adjDic:
            adjDic[aRow[7]] = [(aRow[1], dt.strptime(aRow[0], format_creationdatetime), 
            dt.strptime(aRow[8], format_timestamp), 
            dt.strptime(aRow[9], format_timestamp), 
            dt.strptime(aRow[10], format_timestamp))]
        else:
            adjDic[aRow[7]].append((aRow[1], dt.strptime(aRow[0], format_creationdatetime), 
            dt.strptime(aRow[8], format_timestamp), 
            dt.strptime(aRow[9], format_timestamp), 
            dt.strptime(aRow[10], format_timestamp)))
        #im = Image.fromarray(rgbArray)
        #im.save("C:/Users/kiths/Documents/visualstudiocode-tensorflow/my_file_{0}.png".format(aRow[0]))
    weak, strong = binned(adjDic)

    return rgbDic, coorDic, weak, strong


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


rgb_dic, coord_dic, weak, strong = makeInputs(matrix)

print(weak)
print('---------------------------------------')
print(strong)
print('---------------------------------------')
print(len(weak), len(strong))
eee = 3



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
