import matplotlib.pyplot as plt 
import matplotlib.colorbar as colorbar
import numpy as np 
import preproc
import conf
import datetime



global config

ANUM=166

config = conf.conf()

def map(lon, lat, conf):

    if type(lon) == type(1) and type(lat) == type(1):
        # assume that x,y instead of lon, lat
        return int(lon), int(lat) 

    diff_long = (float(conf.MAX_LONG)-float(conf.MIN_LONG))/(int(conf.STEP)-1)
    diff_lat = (float(conf.MAX_LAT)-float(conf.MIN_LAT))/(int(conf.STEP)-1)

    a = int(round((lon - float(conf.MIN_LONG))/diff_long))
    b = int(round((lat - float(conf.MIN_LAT))/diff_lat))

    if a>=int(conf.STEP) or b>=int(conf.STEP):
        print(lon,lat)
    return a,b


def get_mat(orderdata, bins,  conf):

    '''
    bins = 11 for 1 hour, 22 for 30 mins, 44 for 15 mins, 66 for 10 mins, 132 for 5 mins
    any arbitrary value works well logically, but it's better to use several values above
    '''

    
    initial_time = datetime.datetime(100,1,1,hour=10, minute=0, second=0)
    start_time = initial_time

    p_max_val = -100
    d_max_val = -100

    plist = []
    dlist = []
    lenl = []

    for i in range(bins):
        bin_size = 660.0/float(bins)
        end_time = start_time + datetime.timedelta(minutes= bin_size)
        
        filtered = [a for a in orderdata if start_time.time() <= a[0].time() < end_time.time()]
        #print([a[0].time() for a in filtered])
        mat_pu= np.zeros((conf.STEP,conf.STEP))
        mat_de = np.zeros((conf.STEP,conf.STEP))
        
        for a in filtered:
            """
            a[2,3]: pickup long,lat
            a[5,6]: delivery long,lat
            """

            p_from_i, p_from_j = map(a[2], a[3], conf)
            p_to_i, p_to_j = map(a[5], a[6], conf)
            
            mat_pu[p_from_i,p_from_j] += 1
            mat_de[p_to_i,p_to_j] +=1

        start_time = end_time
        end_time = start_time + datetime.timedelta(minutes= bin_size)

        plist.append(np.rot90(mat_pu))
        dlist.append(np.rot90(mat_de))

        if np.amax(mat_pu) > p_max_val:
            p_max_val = np.amax(mat_pu)
        
        if np.amax(mat_de) > d_max_val:
            d_max_val = np.amax(mat_de)
        
        lenl.append(len(filtered))

    print(lenl)
    return plist, dlist, p_max_val, d_max_val


coord_xy, orderdata = preproc.preproc()

p_list, d_list, p_max, d_max = get_mat(orderdata, 11, config)

rows = len(p_list)

if rows%5==0:
    rows= int(rows/5)
else:
    rows = int(rows//5+1)

fig, axes = plt.subplots(nrows=rows, ncols=5, figsize=(7,rows))

ind = 0
for ax in axes.flat :
    if ind >= len(p_list):
        #break
        #im = ax.imshow()
        pass
    else:
        im = ax.imshow(p_list[ind], cmap='hot', interpolation='nearest', vmax=p_max)
    
    ax.axis('off')    
    ind+=1
    

fig.colorbar(im, ax=axes.ravel().tolist())

#plt.imshow(mat_p, cmap='hot', interpolation='nearest')
#plt.imshow(mat_d[5], cmap='hot', interpolation='nearest', vmax=d_max)

#plt.axis('off')
#plt.colorbar()
plt.show()



