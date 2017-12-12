import conf
import csv
import os
import numpy as np
import random
from PIL import Image
from datetime import datetime as dt
from datetime import timedelta as td
import re

global config
config = conf.conf()

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
        
    dual_picked = get_dual_picked(adjDic)

    sep = separated(matrix, dual_picked)


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

def separated(matrix, dual_picked):
    sepa = []
    twindow = td(seconds=config.TIME_WINDOW*60)
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

def get_dual_picked(adjDic):
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
    twindow = td(seconds=config.TIME_WINDOW*60)
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
    return conf.STEP-processed(lat), processed(lon) 
    for i, j in figure
    """

    if type(lon) == type(1) and type(lat) == type(1):
        # assume that x,y instead of lon, lat
        return int(lon), int(lat) 

    diff_long = (config.MAX_LONG-config.MIN_LONG)/(config.STEP-1)
    diff_lat = (config.MAX_LAT-config.MIN_LAT)/(config.STEP-1)

    a = int(round((lon - config.MIN_LONG)/diff_long))
    b = int(round((lat - config.MIN_LAT)/diff_lat))

    if a>=config.STEP or b>=config.STEP:
        print(lon,lat)
    return a,b

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
    
    random.seed(config.SEED)
    random.shuffle(ret)
    return ret

def updateDicts(req_dic, dem_dic, aRecord):
    p_r1, p_r2 = map(aRecord[0], aRecord[1])
    p_r = '({0},{1})'.format(p_r1, p_r2)
    p_d1, p_d2 = map(aRecord[2], aRecord[3])
    p_d = '({0},{1})'.format(p_d1, p_d2)
    q_r1, q_r2 = map(aRecord[4], aRecord[5])
    q_r = '({0},{1})'.format(q_r1, q_r2)

    if q_r in req_dic[p_r]:
        req_dic[p_r][q_r] += 1
    else:
        req_dic[p_r][q_r] = 1

    if p_d in dem_dic[p_r]:
        dem_dic[p_r][p_d] += 1
    else:
        dem_dic[p_r][p_d] = 1


def getMatAggregation(size, coord_xy):
    mat4req = {}
    mat4dem = {}
    for i in range(size):
        for j in range(size):
            key = '({0},{1})'.format(i,j)
            mat4req[key] = {}
            mat4dem[key] = {}

    for a in coord_xy:
        updateDicts(mat4req, mat4dem, a, config)

    new_mat4req = {k:v for k,v in mat4req.items() if v}
    new_mat4dem = {k:v for k,v in mat4dem.items() if v}

    return new_mat4req, new_mat4dem

def getCoordsFromKey(key):
    return [int(i) for i in re.findall('\d+', key)]

def getCloser(dic, dic_key, closeDist):
    # Assume that closeDist is a positive integer
    # return close key, val and sum(val) of rest(far) keys
    new_dic = {}
    dic_key_coord = getCoordsFromKey(dic_key)
    sumIn = 0
    sumOut = 0
    for key in dic:
        key_coord = getCoordsFromKey(key)
        if (dic_key_coord[0]-closeDist) <= key_coord[0] <= (dic_key_coord[0]+closeDist) and (dic_key_coord[1]-closeDist) <= key_coord[1] <= (dic_key_coord[1]+closeDist):
            new_dic[key] = dic[key]
            sumIn += dic[key]
        else:
            sumOut += dic[key]
    
    return new_dic, sumIn, sumOut

def preproc():
    orderdata = []
    f = open('./data/order_data_2.csv')
    csvReader = csv.reader(f)

    for row in csvReader:
        orderdata.append(row)
    f.close

    # add statistics using part
    coord_dic, sep, dual_picked = makeInputs(orderdata)
    coord_xy = pair_to_xy(coord_dic, sep, dual_picked)
    print('preprocessing complete w/ coordinates inputs')
    return coord_xy

preproc()