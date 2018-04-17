import os
from datetime import datetime as dt
import csv

def get_gap(r1, r2):

    mn = min(r1[1], r2[1])
    mx = max(r1[3], r2[3])

    return mx - mn, r1[3]-r1[1], r2[3]-r2[1]


def parse_date(datetimeStr):
    if type(datetimeStr) is type(dt):
        return datetimeStr
    for fmt in ('%Y-%m-%d %H:%M', '%Y.%m.%d %H:%M' ,'%H:%M:%S'):
        try:
            return dt.strptime(datetimeStr, fmt)
        except ValueError:
            pass
    raise ValueError('No valide date format found for %s'%(datetimeStr))


orderdata = {}
f = open('./data/order_data_2.csv')
csvReader = csv.reader(f)

for row in csvReader:
    dt = parse_date(row[0])
    timeUp = parse_date(row[4])
    timeCatch = parse_date(row[8])
    timePickup = parse_date(row[9])
    timeDeliver = parse_date(row[10])

    row[2] = float(row[2])
    row[3] = float(row[3])
    row[5] = float(row[5])
    row[6] = float(row[6])

    row[0] = dt
    row[4] = dt.replace(hour=timeUp.hour, minute=timeUp.minute, second=timeUp.second)
    row[8] = dt.replace(hour=timeCatch.hour, minute=timeCatch.minute, second=timeCatch.second)
    row[9] = dt.replace(hour=timePickup.hour, minute=timePickup.minute, second=timePickup.second)
    row[10] = dt.replace(hour=timeDeliver.hour, minute=timeDeliver.minute, second=timeDeliver.second)
    
    orderdata[row[1]] = row[4:5] + row[8:]
f.close

search_list = []
for fname in os.listdir('./checking/true_separation/'):

    if fname.endswith('png'):
        fname = fname[:-4]
        fsplit = fname.split('_')

        r1=orderdata[fsplit[0]]
        r2=orderdata[fsplit[1]]

        a,b,c= get_gap(r1,r2)
        print(fsplit, a,b,c)