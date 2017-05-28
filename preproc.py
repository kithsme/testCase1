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
GRAY_SCALE = 45
STEP = 32
TIME_WINDOW = 16
DRAW_FIGURE = False
SEED = 1120

def makeInputs(matrix):
    
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
        
    dual_picked = get_dual_picked(adjDic, TIME_WINDOW)

    sep = separated(matrix, dual_picked, tw=TIME_WINDOW)

    return coorDic, sep, dual_picked

def parse_date(datetimeStr):
    if type(datetimeStr) is type(dt):
        return datetimeStr
    for fmt in ('%Y-%m-%d %H:%M', '%Y.%m.%d %H:%M' ,'%H:%M:%S'):
        try:
            return dt.strptime(datetimeStr, fmt)
        except ValueError:
            pass
    raise ValueError('No valide date format found for %s'%(datetimeStr))

def separated(matrix, dual_picked, tw=15):
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
                exist = sum(1 for x in dual_picked if x[0]==aId and x[1]==bId)
                if exist == 0:
                    sepa.append( (aId,bId) )
            else:
                break
    return sepa

def get_dual_picked(adjDic, tw=15):
    """
    return tuples (order1_id, order2_id, dual_picked_type) 
        # type 1 : pick1 pick2 del1 del2
        # type 2 : pick2 pick1 del1 del2
        # type 3 : pick1 pick2 del2 del1
        # type 4 : pick2 pick1 del2 del1
    a[0]: order id
    a[1]: creation timestamp
    a[2,3,4]: catched timestamp, pickedup timestamp, delivered timestamp
    """
    type1_cnt = 0
    type2_cnt = 0
    type3_cnt = 0
    type4_cnt = 0
    dual_picked=[]
    twindow = td(seconds=tw*60)
    prev=None
    for key in adjDic:  # for every rider
        prev = adjDic[key][0] 
        for a in adjDic[key]:
            if a[2] > prev[2] + twindow:
                pass
            elif prev[3] < a[3] < prev[4] < a[4]: # type 1
                dual_picked.append((prev[0], a[0], 1))
                type1_cnt +=1
            elif a[3] < prev[3] < prev[4] < a[4]: # type 2
                dual_picked.append((prev[0], a[0], 2))
                type2_cnt +=1
            elif prev[3] < a[3] < a[4] < prev[4]: # type 3
                dual_picked.append((prev[0], a[0], 3))
                type3_cnt +=1
            elif a[3] < prev[3] < a[4] < prev[4]: # type 4
                dual_picked.append((prev[0], a[0], 4))
                type4_cnt +=1

            prev = a # order update and next
    
    print('Total {0} duals, 1: {1}, 2: {2}, 3: {3}, 4: {4}'.format(type1_cnt+type2_cnt+type3_cnt+type4_cnt, type1_cnt, type2_cnt, type3_cnt, type4_cnt))
    return dual_picked

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
    x,y=[],[]
    count = 0
    for a in xy:
        """
        a[0,1]: previous from long,lat
        a[2,3]: previous to long,lat
        a[4,5]: following from long,lat
        a[6,7]: following to long,lat
        a[8]: type (0~4)
        a[9,10]: prev order id , following order id 
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
        
        points = [(p_from_i, p_from_j), (p_to_i, p_to_j), (q_from_i, q_from_j), (q_to_i, q_to_j)]
        
        # if two orders share a point, previous one always overwrite it
        fill_point(rgb_array, q_from_i, q_from_j, 255,0,0)
        fill_point(rgb_array, q_to_i, q_to_j, 0, 0, 255)
        fill_point(rgb_array, p_from_i, p_from_j, 255, 255, 0)
        fill_point(rgb_array, p_to_i, p_to_j, 0, 255, 255)
        
        x_this, y_this = generate_cases(rgb_array, points, a[8], a[9], a[10], figure=DRAW_FIGURE)
        
        x += x_this
        y += y_this
    
    print('Total generaged case = {0}'.format(len(x)))
    return x,y

def fill_line(target_array, line, r1,g1,b1, r2,g2,b2):
    line_half = len(line)//2+1
    for l in line[:line_half]:
        fill_point(target_array, l[0], l[1], r1,g1,b1)
    for l in line[line_half:]:
        fill_point(target_array, l[0], l[1], r2,g2,b2)
    
def generate_array(original_array, pts, form):
    pfrom=pts[0]
    pto=pts[1]
    qfrom=pts[2]
    qto=pts[3]
    tmp_rgb_ary = np.empty_like(original_array)
    tmp_rgb_ary[:] = original_array
    if form==0:
        line1 = line(pfrom[0], pfrom[1], pto[0], pto[1])
        line2 = line(qfrom[0], qfrom[1], qto[0], qto[1])

        fill_line(tmp_rgb_ary, line1, 150,150,150, 150,150,150)
        fill_line(tmp_rgb_ary, line2, 150,150,150, 150,150,150)
    elif form==1:
        line1 = line(pfrom[0], pfrom[1], qfrom[0], qfrom[1])
        line2 = line(qfrom[0], qfrom[1], pto[0], pto[1])
        line3 = line(pto[0], pto[1], qto[0], qto[1])

        fill_line(tmp_rgb_ary, line1, 150,150,0, 150,150,0)
        fill_line(tmp_rgb_ary, line2, 150,0,0, 150,0,0)
        fill_line(tmp_rgb_ary, line3, 0,150,150, 0,150,150)
    elif form==2:
        line1 = line(qfrom[0], qfrom[1], pfrom[0], pfrom[1])
        line2 = line(pfrom[0], pfrom[1], pto[0], pto[1])
        line3 = line(pto[0], pto[1], qto[0], qto[1])

        fill_line(tmp_rgb_ary, line1, 150,0,0, 150,0,0)
        fill_line(tmp_rgb_ary, line2, 150,150,0, 150,150,0)
        fill_line(tmp_rgb_ary, line3, 0,150,150, 0,150,150)
    elif form==3:
        line1 = line(pfrom[0], pfrom[1], qfrom[0], qfrom[1])
        line2 = line(qfrom[0], qfrom[1], qto[0], qto[1])
        line3 = line(qto[0], qto[1], pto[0], pto[1])

        fill_line(tmp_rgb_ary, line1, 150,150,0, 150,150,0)
        fill_line(tmp_rgb_ary, line2, 150,0,0, 150,0,0)
        fill_line(tmp_rgb_ary, line3, 0,0,150, 0,0,150)
    elif form==4:
        line1 = line(qfrom[0], qfrom[1], pfrom[0], pfrom[1])
        line2 = line(pfrom[0], pfrom[1], qto[0], qto[1])
        line3 = line(qto[0], qto[1], pto[0], pto[1])

        fill_line(tmp_rgb_ary, line1, 150,0,0, 150,0,0)
        fill_line(tmp_rgb_ary, line2, 150,150,0, 150,150,0)
        fill_line(tmp_rgb_ary, line3, 0,0,150, 0,0,150)
    return tmp_rgb_ary

def generate_cases(original_array, pts, seq, aId, bId, figure=False):
    """
    pts = [(p_from_i, p_from_j), (p_to_i, p_to_j), (q_from_i, q_from_j), (q_to_i, q_to_j)]
    """
    pfrom=pts[0]
    pto=pts[1]
    qfrom=pts[2]
    qto=pts[3]
    x,y = [], []
    if seq == 0: # separated, link pfrom-pto, qfrom-qto
        tmp_rgb_ary = generate_array(original_array, pts, 0)
        if figure:
            im = Image.fromarray(tmp_rgb_ary)
            im.save("./result/fig/{0}_fig_{1}_{2}_{3}pix_{4}gray.png".format('right_sep', aId, bId, STEP, GRAY_SCALE))
        x.append(tmp_rgb_ary.reshape(STEP*STEP*3).tolist())
        y.append([1,0])

        tmp_rgb_ary = generate_array(original_array, pts, 1)
        if figure:
            im = Image.fromarray(tmp_rgb_ary)
            im.save("./result/fig/{0}_fig_{1}_{2}_{3}pix_{4}gray.png".format('wrong_dual', aId, bId, STEP, GRAY_SCALE))
        x.append(tmp_rgb_ary.reshape(STEP*STEP*3).tolist())
        y.append([0,1])
    elif seq == 1: # type 1, link pfrom-qfrom-pto-qto
        tmp_rgb_ary = generate_array(original_array, pts, 1)
        if figure:
            im = Image.fromarray(tmp_rgb_ary)
            im.save("./result/fig/{0}_fig_{1}_{2}_{3}pix_{4}gray.png".format('right_dual', aId, bId, STEP, GRAY_SCALE))
        x.append(tmp_rgb_ary.reshape(STEP*STEP*3).tolist())
        y.append([1,0])

        tmp_rgb_ary = generate_array(original_array, pts, 0)
        if figure:
            im = Image.fromarray(tmp_rgb_ary)
            im.save("./result/fig/{0}_fig_{1}_{2}_{3}pix_{4}gray.png".format('wrong_sep', aId, bId, STEP, GRAY_SCALE))
        x.append(tmp_rgb_ary.reshape(STEP*STEP*3).tolist())
        y.append([0,1])
    elif seq == 2: # type 2, link qfrom-pfrom-pto-qto
        tmp_rgb_ary = generate_array(original_array, pts, 2)
        if figure:
            im = Image.fromarray(tmp_rgb_ary)
            im.save("./result/fig/{0}_fig_{1}_{2}_{3}pix_{4}gray.png".format('right_dual', aId, bId, STEP, GRAY_SCALE))
        x.append(tmp_rgb_ary.reshape(STEP*STEP*3).tolist())
        y.append([1,0])

        tmp_rgb_ary = generate_array(original_array, pts, 0)
        if figure:
            im = Image.fromarray(tmp_rgb_ary)
            im.save("./result/fig/{0}_fig_{1}_{2}_{3}pix_{4}gray.png".format('wrong_sep', aId, bId, STEP, GRAY_SCALE))
        x.append(tmp_rgb_ary.reshape(STEP*STEP*3).tolist())
        y.append([0,1])

    elif seq == 3:
        tmp_rgb_ary = generate_array(original_array, pts, 3)
        if figure:
            im = Image.fromarray(tmp_rgb_ary)
            im.save("./result/fig/{0}_fig_{1}_{2}_{3}pix_{4}gray.png".format('right_dual', aId, bId, STEP, GRAY_SCALE))
        x.append(tmp_rgb_ary.reshape(STEP*STEP*3).tolist())
        y.append([1,0])

        tmp_rgb_ary = generate_array(original_array, pts, 0)
        if figure:
            im = Image.fromarray(tmp_rgb_ary)
            im.save("./result/fig/{0}_fig_{1}_{2}_{3}pix_{4}gray.png".format('wrong_sep', aId, bId, STEP, GRAY_SCALE))
        x.append(tmp_rgb_ary.reshape(STEP*STEP*3).tolist())
        y.append([0,1])
    elif seq==4:
        tmp_rgb_ary = generate_array(original_array, pts, 4)
        if figure:
            im = Image.fromarray(tmp_rgb_ary)
            im.save("./result/fig/{0}_fig_{1}_{2}_{3}pix_{4}gray.png".format('right_dual', aId, bId, STEP, GRAY_SCALE))
        x.append(tmp_rgb_ary.reshape(STEP*STEP*3).tolist())
        y.append([1,0])

        tmp_rgb_ary = generate_array(original_array, pts, 0)
        if figure:
            im = Image.fromarray(tmp_rgb_ary)
            im.save("./result/fig/{0}_fig_{1}_{2}_{3}pix_{4}gray.png".format('wrong_sep', aId, bId, STEP, GRAY_SCALE))
        x.append(tmp_rgb_ary.reshape(STEP*STEP*3).tolist())
        y.append([0,1])

    return x,y

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

def fill_point(nda,i,j,r,g,b):
    a = nda[STEP-j][i]
    if (a[0] == GRAY_SCALE or a[0]==0) and (a[1] == GRAY_SCALE or a[1]==0) and (a[2] == GRAY_SCALE or a[2]==0):
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

def pair_to_xy(dic, sep, dual_picked):
    ret = []
    for pair in sep:
        a, b = pair[0], pair[1]
        y= [0]
        x = dic[a] + dic[b] + y + [a,b]
        ret.append(x)
    
    for pair in dual_picked:
        a, b = pair[0], pair[1]
        y= [pair[2]]
        x = dic[a] + dic[b] + y + [a,b]
        ret.append(x)
    
    random.seed(SEED)
    random.shuffle(ret)
    return ret

def draw_map(xy):
    rgb_array_map = np.zeros((STEP,STEP,3), 'uint8')
    for a in xy:
        q_to_i, q_to_j = map(a[6], a[7])
        fill_point(rgb_array_map, q_to_i, q_to_j, GRAY_SCALE, GRAY_SCALE, GRAY_SCALE)
    
    if not os.path.exists("./result/fig"):
        os.makedirs("./result/fig")
    im = Image.fromarray(rgb_array_map)
    im.save("./result/fig/whole_fig.png")

    return rgb_array_map

def preproc(rgb=0):
    orderdata = []
    f = open('./data/order_data_2.csv')
    csvReader = csv.reader(f)

    for row in csvReader:
        orderdata.append(row)
    f.close

    coord_dic, sep, dual_picked = makeInputs(orderdata)
    # print( len(sep), len(weak), len(strong) )  #check for values

    coord_xy = pair_to_xy(coord_dic, sep, dual_picked)
    if rgb==1:
        gray_map = draw_map(coord_xy)
        rgb_x,rgb_y = xy_to_rgbArray(coord_xy, gray_map)
        print('preprocessing complete w/ rgb inputs')
        return rgb_x, rgb_y
    else:
        print('preprocessing complete w/ coordinates inputs')
        return coord_xy