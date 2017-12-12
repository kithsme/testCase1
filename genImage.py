import csv
import os
from PIL import Image
from datetime import datetime as dt
from datetime import timedelta as td
import re
import numpy as np

def getConfig(config_id):
    f = open('./model/{0}/preproc_config.csv'.format(config_id), 'r')
    reader = csv.reader(f, delimiter=',', quotechar='|')
    row = None
    for a in reader:
        row = a
    f.close()
    return row

def map(lon, lat, conf):
    """
    get longitude and latitude, 
    return int(conf[5])-processed(lat), processed(lon) 
    for i, j in figure
    """

    if type(lon) == type(1) and type(lat) == type(1):
        # assume that x,y instead of lon, lat
        return int(lon), int(lat) 

    diff_long = (float(conf[1])-float(conf[0]))/(int(conf[5])-1)
    diff_lat = (float(conf[3])-float(conf[2]))/(int(conf[5])-1)

    a = int(round((lon - float(conf[0]))/diff_long))
    b = int(round((lat - float(conf[2]))/diff_lat))

    if a>=int(conf[5]) or b>=int(conf[5]):
        print(lon,lat)
    return a,b

def draw_map(xy, conf):
    rgb_array_map = np.zeros((int(conf[5]),int(conf[5]),3), 'uint8')
    for a in xy:
        q_to_i, q_to_j = map(a[6], a[7], conf)
        fill_point(rgb_array_map, q_to_i, q_to_j, int(conf[4]), int(conf[4]), int(conf[4]), conf)
    
    if not os.path.exists("./result/fig"):
        os.makedirs("./result/fig")
    im = Image.fromarray(rgb_array_map)
    im.save("./result/fig/whole_fig.png")

    return rgb_array_map

def getHitmap(windowed_x, pdtype='d', conf):
    hit_map = np.zeros((int(conf[5]),int(conf[5])), 'uint8')
    for a in windowed_x:
        i= -1
        j= -1
        if pdtype=='p':
            i,j = map(a[4],a[5], conf)
        else:
            i, j = map(a[6], a[7], conf)
        inc_point(hit_map, i,j,conf)
    
    return hit_map


def xy_to_rgbArray(xy, gray_map, conf):
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
        p_from_i, p_from_j = map(a[0], a[1], conf)
        p_to_i, p_to_j = map(a[2], a[3], conf)
        #p_line = line(p_from_i, p_from_j, p_to_i, p_to_j)

        q_from_i, q_from_j = map(a[4], a[5], conf)
        q_to_i, q_to_j = map(a[6], a[7], conf)
        #q_line = line(q_from_i, q_from_j, q_to_i, q_to_j)
        
        points = [(p_from_i, p_from_j), (p_to_i, p_to_j), (q_from_i, q_from_j), (q_to_i, q_to_j)]
        
        # if two orders share a point, previous one always overwrite it
        fill_point(rgb_array, q_from_i, q_from_j, 255,0,0, conf)
        fill_point(rgb_array, q_to_i, q_to_j, 0, 0, 255, conf)
        fill_point(rgb_array, p_from_i, p_from_j, 255, 255, 0, conf)
        fill_point(rgb_array, p_to_i, p_to_j, 0, 255, 255, conf)
        
        x_this, y_this = generate_cases(rgb_array, points, a[8], a[9], a[10], conf, figure=False)
        x += x_this
        y += y_this
    
    #print('Total generaged case = {0}'.format(len(x)))
    return x,y

def fill_line(target_array, line, r1,g1,b1, r2,g2,b2, conf):
    line_half = len(line)//2+1
    for l in line[:line_half]:
        fill_point(target_array, l[0], l[1], r1,g1,b1, conf)
    for l in line[line_half:]:
        fill_point(target_array, l[0], l[1], r2,g2,b2, conf)
    
def generate_array(original_array, pts, form, conf):
    pfrom=pts[0]
    pto=pts[1]
    qfrom=pts[2]
    qto=pts[3]
    tmp_rgb_ary = np.empty_like(original_array)
    tmp_rgb_ary[:] = original_array
    if form==0:
        line1 = line(pfrom[0], pfrom[1], pto[0], pto[1])
        line2 = line(qfrom[0], qfrom[1], qto[0], qto[1])

        fill_line(tmp_rgb_ary, line1, 150,150,150, 150,150,150, conf)
        fill_line(tmp_rgb_ary, line2, 150,150,150, 150,150,150, conf)
    elif form==1:
        line1 = line(pfrom[0], pfrom[1], qfrom[0], qfrom[1])
        line2 = line(qfrom[0], qfrom[1], pto[0], pto[1])
        line3 = line(pto[0], pto[1], qto[0], qto[1])

        fill_line(tmp_rgb_ary, line1, 150,150,0, 150,150,0, conf)
        fill_line(tmp_rgb_ary, line2, 150,0,0, 150,0,0, conf)
        fill_line(tmp_rgb_ary, line3, 0,150,150, 0,150,150, conf)
    elif form==2:
        line1 = line(qfrom[0], qfrom[1], pfrom[0], pfrom[1])
        line2 = line(pfrom[0], pfrom[1], pto[0], pto[1])
        line3 = line(pto[0], pto[1], qto[0], qto[1])

        fill_line(tmp_rgb_ary, line1, 150,0,0, 150,0,0, conf)
        fill_line(tmp_rgb_ary, line2, 150,150,0, 150,150,0, conf)
        fill_line(tmp_rgb_ary, line3, 0,150,150, 0,150,150, conf)
    elif form==3:
        line1 = line(pfrom[0], pfrom[1], qfrom[0], qfrom[1])
        line2 = line(qfrom[0], qfrom[1], qto[0], qto[1])
        line3 = line(qto[0], qto[1], pto[0], pto[1])

        fill_line(tmp_rgb_ary, line1, 150,150,0, 150,150,0, conf)
        fill_line(tmp_rgb_ary, line2, 150,0,0, 150,0,0, conf)
        fill_line(tmp_rgb_ary, line3, 0,0,150, 0,0,150, conf)
    elif form==4:
        line1 = line(qfrom[0], qfrom[1], pfrom[0], pfrom[1])
        line2 = line(pfrom[0], pfrom[1], qto[0], qto[1])
        line3 = line(qto[0], qto[1], pto[0], pto[1])

        fill_line(tmp_rgb_ary, line1, 150,0,0, 150,0,0, conf)
        fill_line(tmp_rgb_ary, line2, 150,150,0, 150,150,0, conf)
        fill_line(tmp_rgb_ary, line3, 0,0,150, 0,0,150, conf)
    
    return tmp_rgb_ary

def generate_cases(original_array, pts, seq, aId, bId, conf, figure=False, figdir="./result/fig/"):
    """
    pts = [(p_from_i, p_from_j), (p_to_i, p_to_j), (q_from_i, q_from_j), (q_to_i, q_to_j)]
    """
    pfrom=pts[0]
    pto=pts[1]
    qfrom=pts[2]
    qto=pts[3]
    x,y = [], []
    if seq == 0: # separated, link pfrom-pto, qfrom-qto
        tmp_rgb_ary = generate_array(original_array, pts, 0, conf)
        if figure:
            im = Image.fromarray(tmp_rgb_ary)
            im.save("{0}{1}_fig_{2}_{3}_{4}pix_{5}gray.png".format(figdir,'right_sep', aId, bId, int(conf[5]), int(conf[4])))
        x.append(tmp_rgb_ary.reshape(int(conf[5])*int(conf[5])*3).tolist())
        y.append([1,0])

        
        tmp_rgb_ary = generate_array(original_array, pts, 1, conf)
        if figure:
            im = Image.fromarray(tmp_rgb_ary)
            im.save("{0}{1}_fig_{2}_{3}_{4}pix_{5}gray.png".format(figdir, 'wrong_dual_1', aId, bId, int(conf[5]), int(conf[4])))
        x.append(tmp_rgb_ary.reshape(int(conf[5])*int(conf[5])*3).tolist())
        y.append([0,1])
        '''
        tmp_rgb_ary = generate_array(original_array, pts, 2, conf)
        if figure:
            im = Image.fromarray(tmp_rgb_ary)
            im.save("{0}{1}_fig_{2}_{3}_{4}pix_{5}gray.png".format(figdir, 'wrong_dual_2', aId, bId, int(conf[5]), int(conf[4])))
        x.append(tmp_rgb_ary.reshape(int(conf[5])*int(conf[5])*3).tolist())
        y.append([0,1])

        tmp_rgb_ary = generate_array(original_array, pts, 3, conf)
        if figure:
            im = Image.fromarray(tmp_rgb_ary)
            im.save("{0}{1}_fig_{2}_{3}_{4}pix_{5}gray.png".format(figdir, 'wrong_dual_3', aId, bId, int(conf[5]), int(conf[4])))
        x.append(tmp_rgb_ary.reshape(int(conf[5])*int(conf[5])*3).tolist())
        y.append([0,1])

        tmp_rgb_ary = generate_array(original_array, pts, 4, conf)
        if figure:
            im = Image.fromarray(tmp_rgb_ary)
            im.save("{0}{1}_fig_{2}_{3}_{4}pix_{5}gray.png".format(figdir, 'wrong_dual_4', aId, bId, int(conf[5]), int(conf[4])))
        x.append(tmp_rgb_ary.reshape(int(conf[5])*int(conf[5])*3).tolist())
        y.append([0,1])
        '''
    elif seq == 1: # type 1, link pfrom-qfrom-pto-qto
        tmp_rgb_ary = generate_array(original_array, pts, 1, conf)
        if figure:
            im = Image.fromarray(tmp_rgb_ary)
            im.save("{0}{1}_fig_{2}_{3}_{4}pix_{5}gray.png".format(figdir, 'right_dual', aId, bId, int(conf[5]), int(conf[4])))
        x.append(tmp_rgb_ary.reshape(int(conf[5])*int(conf[5])*3).tolist())
        y.append([1,0])
        '''
        tmp_rgb_ary = generate_array(original_array, pts, 0, conf)
        if figure:
            im = Image.fromarray(tmp_rgb_ary)
            im.save("{0}{1}_fig_{2}_{3}_{4}pix_{5}gray.png".format(figdir, 'wrong_sep', aId, bId, int(conf[5]), int(conf[4])))
        x.append(tmp_rgb_ary.reshape(int(conf[5])*int(conf[5])*3).tolist())
        y.append([0,1])
        '''
    elif seq == 2: # type 2, link qfrom-pfrom-pto-qto
        tmp_rgb_ary = generate_array(original_array, pts, 2, conf)
        if figure:
            im = Image.fromarray(tmp_rgb_ary)
            im.save("{0}{1}_fig_{2}_{3}_{4}pix_{5}gray.png".format(figdir,'right_dual', aId, bId, int(conf[5]), int(conf[4])))
        x.append(tmp_rgb_ary.reshape(int(conf[5])*int(conf[5])*3).tolist())
        y.append([1,0])
        '''
        tmp_rgb_ary = generate_array(original_array, pts, 0, conf)
        if figure:
            im = Image.fromarray(tmp_rgb_ary)
            im.save("{0}{1}_fig_{2}_{3}_{4}pix_{5}gray.png".format(figdir, 'wrong_sep', aId, bId, int(conf[5]), int(conf[4])))
        x.append(tmp_rgb_ary.reshape(int(conf[5])*int(conf[5])*3).tolist())
        y.append([0,1])
        '''
    elif seq == 3:
        tmp_rgb_ary = generate_array(original_array, pts, 3, conf)
        if figure:
            im = Image.fromarray(tmp_rgb_ary)
            im.save("{0}{1}_fig_{2}_{3}_{4}pix_{5}gray.png".format(figdir, 'right_dual', aId, bId, int(conf[5]), int(conf[4])))
        x.append(tmp_rgb_ary.reshape(int(conf[5])*int(conf[5])*3).tolist())
        y.append([1,0])
        '''
        tmp_rgb_ary = generate_array(original_array, pts, 0, conf)
        if figure:
            im = Image.fromarray(tmp_rgb_ary)
            im.save("{0}{1}_fig_{2}_{3}_{4}pix_{5}gray.png".format(figdir, 'wrong_sep', aId, bId, int(conf[5]), int(conf[4])))
        x.append(tmp_rgb_ary.reshape(int(conf[5])*int(conf[5])*3).tolist())
        y.append([0,1])
        '''
    elif seq==4:
        tmp_rgb_ary = generate_array(original_array, pts, 4, conf)
        if figure:
            im = Image.fromarray(tmp_rgb_ary)
            im.save("{0}{1}_fig_{2}_{3}_{4}pix_{5}gray.png".format(figdir, 'right_dual', aId, bId, int(conf[5]), int(conf[4])))
        x.append(tmp_rgb_ary.reshape(int(conf[5])*int(conf[5])*3).tolist())
        y.append([1,0])
        '''
        tmp_rgb_ary = generate_array(original_array, pts, 0, conf)
        if figure:
            im = Image.fromarray(tmp_rgb_ary)
            im.save("{0}{1}_fig_{2}_{3}_{4}pix_{5}gray.png".format(figdir, 'wrong_sep', aId, bId, int(conf[5]), int(conf[4])))
        x.append(tmp_rgb_ary.reshape(int(conf[5])*int(conf[5])*3).tolist())
        y.append([0,1])
        '''
    return x,y

def fig_test(rgb_array, conf):
    pi, pj = 1, 1
    qi, qj = 10, 30
    except_point = [(pi,pj),(qi,qj)]
    pqline = line(pi,pj,qi,qj)
    for l in pqline:
        if not l in except_point:
            fill_point(rgb_array, l[0], l[1], 255,255,255, conf)
    fill_point(rgb_array, pi, pj, 255,0,0)
    fill_point(rgb_array, qi,qj,0,0,255)

    im = Image.fromarray(rgb_array)
    im.save("./result/fig/test_fig.png")



def fill_point(nda,i,j,r,g,b, conf):
    a = nda[int(conf[5])-j][i]
    if (a[0] == int(conf[4]) or a[0]==0) and (a[1] == int(conf[4]) or a[1]==0) and (a[2] == int(conf[4]) or a[2]==0):
        nda[int(conf[5])-j][i][0] = r
        nda[int(conf[5])-j][i][1] = g
        nda[int(conf[5])-j][i][2] = b

def inc_point(nda,i,j, conf):
    a = nda[int(conf[5])-j][i]
    if (a[0] == int(conf[4]) or a[0]==0) and (a[1] == int(conf[4]) or a[1]==0) and (a[2] == int(conf[4]) or a[2]==0):
        nda[int(conf[5])-j][i] += 1

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
