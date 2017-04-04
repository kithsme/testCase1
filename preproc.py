import csv
import os
import numpy as np
import random
from PIL import Image
from datetime import datetime as dt
from datetime import timedelta as td

MIN_LONG = 127.07
MAX_LONG = 127.14
MIN_LAT = 37.48
MAX_LAT = 37.53
GRAY_SCALE = 15
STEP = 64
TIME_WINDOW = 1.5
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
            if prev[2]<a[2]<prev[3] and (a[3] < prev[4] or a[3] < prev[3]):
                strong.append( [prev[0], a[0]] )
            elif prev[2]<a[2] and a[2] < prev[4]:
                weak.append( [prev[0], a[0]] )
            prev = a

    return weak, strong

def map(lon, lat):
    """
    get longitude and latitude, 
    return STEP-processed(lat), processed(lon) 
    for i, j in figure
    """
    diff_long = (MAX_LONG-MIN_LONG)/(STEP-1)
    diff_lat = (MAX_LAT-MIN_LAT)/(STEP-1)

    a = int(round((lon - MIN_LONG)/diff_long))
    b = int(round((lat - MIN_LAT)/diff_lat))

    if a>=STEP or b>=STEP:
        print(lon,lat)
    return a,b

def xy_to_rgbArray(xy, gray_map):
    x,y,x_t,x_f=[],[],[],[]
    count = 0
    for a in xy:
        """
        a[0,1]: previous from long,lat
        a[2,3]: previous to long,lat
        a[4,5]: following from long,lat
        a[6,7]: following to long,lat
        a[8,9]: y (1 or 0)
        a[10,11]: prev order id , following order id 
        """
        # i, j index is reverted b/c it is stored to array
        rgb_array = np.empty_like(gray_map)
        rgb_array[:] = gray_map
        #fig_test(rgb_array)
        p_from_i, p_from_j = map(a[0], a[1])
        p_to_i, p_to_j = map(a[2], a[3])
        #p_line = line(p_from_i, p_from_j, p_to_i, p_to_j)

        q_from_i, q_from_j = map(a[4], a[5])
        q_to_i, q_to_j = map(a[6], a[7])
        #q_line = line(q_from_i, q_from_j, q_to_i, q_to_j)
        
        except_point = [(p_from_i, p_from_j), (p_to_i, p_to_j), (q_from_i, q_from_j), (q_to_i, q_to_j)]

        p_line = line(p_from_i, p_from_j, q_from_i, q_from_j)
        p_half = len(p_line)//2+1
        q_line = line(p_to_i, p_to_j, q_to_i, q_to_j)
        q_half = len(q_line)//2+1
        r_line = line(q_from_i, q_from_j, p_to_i, p_to_j)
        r_half = len(r_line)//2+1

        # if two orders share a point, previous one always overwrite it
        fill_point(rgb_array, q_from_i, q_from_j, 255,0,0)
        fill_point(rgb_array, q_to_i, q_to_j, 0, 0, 255)
        fill_point(rgb_array, p_from_i, p_from_j, 255, 255, 0)
        fill_point(rgb_array, p_to_i, p_to_j, 0, 255, 255)

        for l in q_line[:q_half]:
            if not l in except_point:
                fill_point(rgb_array, l[0], l[1], 0,150,150)
        for l in q_line[q_half:]:
            if not l in except_point:
                fill_point(rgb_array, l[0], l[1], 0,0,150)

        for l in r_line[:r_half]:
            if not l in except_point:
                fill_point(rgb_array, l[0], l[1], 150,0,0)
        for l in r_line[r_half:]:
            if not l in except_point:
                fill_point(rgb_array, l[0], l[1], 0,150,150)

        for l in p_line[:p_half]:
            if not l in except_point:
                fill_point(rgb_array, l[0], l[1], 150,150,0)
        for l in p_line[p_half:]:
            if not l in except_point:
                fill_point(rgb_array, l[0], l[1], 150,0,0)
        '''
        for l in q_line[:q_half]:
            if not l in except_point:
                fill_point(rgb_array, l[0], l[1], 150,0,0)
        
        for l in q_line[q_half:]:
            if not l in except_point:
                fill_point(rgb_array, l[0], l[1], 0,0,150)

        for l in p_line[:p_half]:
            if not l in except_point:
                fill_point(rgb_array, l[0], l[1], 150,150,0)
        
        for l in p_line[p_half:]:
            if not l in except_point:
                fill_point(rgb_array, l[0], l[1], 0,150,150)
        '''

        x.append(rgb_array.reshape(STEP*STEP*3).tolist())
        y.append(a[8:10])

        if a[8]==1:
            x_t.append(rgb_array.reshape(STEP*STEP*3).tolist())
        else:
            x_f.append(rgb_array.reshape(STEP*STEP*3).tolist())
        
        count +=1

        ## print for debugging and illustrative examples
        if count % 100 == 0:
            im = Image.fromarray(rgb_array)
            im.save("./result/fig/{0}_fig_{1}_{2}.png".format(a[8], a[10], a[11]))
        
    return x,y, x_t, x_f

def fig_test(rgb_array):
    pi, pj = 1, 1
    qi, qj = 10, 30
    except_point = [(pi,pj),(qi,qj)]
    pqline = line(pi,pj,qi,qj)
    for l in pqline:
        if not l in except_point:
            fill_point(rgb_array, l[0], l[1], 255,255,255)
    fill_point(rgb_array, pi, pj, 255,0,0)
    fill_point(rgb_array, qi,qj,0,0,255)

    im = Image.fromarray(rgb_array)
    im.save("./result/fig/test_fig.png")

def fill_point(nda, i,j,r,g,b):
    nda[STEP-j][i][0] = r
    nda[STEP-j][i][1] = g
    nda[STEP-j][i][2] = b

def line(i1,j1,i2, j2):
    x_axis_len = abs(i1-i2)
    y_axis_len = abs(j1-j2)
    x_diff = i1-i2
    y_diff = j1-j2

    #print(i1, j1, i2, j2)
    min_axis_len = min(x_axis_len, y_axis_len)
    max_axis_len = max(x_axis_len, y_axis_len)
    min_axis = 'y' if min_axis_len==y_axis_len else 'x'

    ret = []
    x_dir = 1
    y_dir = 1
    if x_diff == 0:
        x_dir = 0
    elif x_diff > 0:
        x_dir = -1
    
    if y_diff == 0:
        y_dir = 0
    elif y_diff > 0:
        y_dir = -1

    xs = []
    ys = []

    for i in range(max_axis_len):
        level = int(i//((max_axis_len+1)/(min_axis_len+1)))
        if x_axis_len == y_axis_len:
            xs.append(i1 + i * x_dir)
            ys.append(j1 + i * y_dir)
        elif min_axis == 'y':
            xs.append(i1 + i * x_dir)
            ys.append(j1 + level * y_dir)
        elif min_axis =='x':
            ys.append(j1 + i*y_dir)
            xs.append(i1 + level * x_dir)            
    
    for i in range(max_axis_len):
        a,b = xs[i],ys[i]
        ret.append((a,b))

    return ret

def pair_to_xy(dic, sep, weak, strong, mode='w'):

    ret = []
    for pair in sep:
        a, b = pair[0], pair[1]
        y=[0,1] # bc it comes from sep, y = 0
        x = dic[a] + dic[b] + y + [a,b]
        ret.append(x)
    
    for pair in strong:
        a, b = pair[0], pair[1]
        y=[1,0] # bc it comes from strong binned, y = 1
        x = dic[a] + dic[b] + y + [a,b]
        ret.append(x)

    if mode=='w':
        for pair in weak:
            a, b = pair[0], pair[1]
            y=[1,0] # bc it comes from weak binned, y = 1
            x = dic[a] + dic[b] + y + [a,b]
            ret.append(x)
    else:
        pass
        #for pair in weak:
        #    a, b = pair[0], pair[1]
        #    y=[0,1] # bc it comes from weak binned, but policy is strong so y = 0
        #    x = dic[a] + dic[b] + y
        #    ret.append(x)
    
    random.shuffle(ret)
    return ret

def draw_map(xy):
    rgb_array_map = np.zeros((STEP,STEP,3), 'uint8')
    for a in xy:
        q_to_i, q_to_j = map(a[6], a[7])
        fill_point(rgb_array_map, q_to_i, q_to_j, GRAY_SCALE, GRAY_SCALE, GRAY_SCALE)
        
    im = Image.fromarray(rgb_array_map)
    im.save("./result/fig/whole_fig.png")

    return rgb_array_map

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
    f = open('./data/order_data_2.csv')
    csvReader = csv.reader(f)

    for row in csvReader:
        orderdata.append(row)
    f.close

    coord_dic, sep, weak, strong = makeInputs(orderdata)
    # print( len(sep), len(weak), len(strong) )  #check for values

    coord_xy = pair_to_xy(coord_dic, sep, weak, strong, mode=SEPATED_MODE)
    if rgb==1:
        gray_map = draw_map(coord_xy)
        rgb_x,rgb_y, x_t, x_f = xy_to_rgbArray(coord_xy, gray_map)
        print('preprocessing complete w/ rgb inputs: interest/total {0}/{1}'.format(len(x_t), len(rgb_x)))
        return rgb_x, rgb_y, x_t, x_f
    else:
        print('preprocessing complete w/ coordinates inputs')
        write_csv(coord_xy, rgb)
