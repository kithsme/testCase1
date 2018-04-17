import folium
import os
import time
from selenium import webdriver


orderdata = []
f = open('coords_xy.csv')
csvReader = csv.reader(f)
webdriverDir= 'C:/Users/pos/chromedriver_win32/chromedriver.exe'
delay=1


for row in csvReader:

    aPickLong = float(row[0])
    aPickLat = float(row[1])
    aDelLong = float(row[2])
    aDelLat = float(row[3])
    bPickLong = float(row[4])
    bPickLat = float(row[5])
    bDelLong = float(row[6])
    bDelLat = float(row[7])
    seq = int(row[8])
    reqId = row[9] +'_'+ row[10]

    m = folium.Map(location=[37.505, 127.105], tiles='Stamen Toner')
    m.fit_bounds([[37.4945, 127.0793], [37.5204, 127.1259]])


    if seq == 0:
        folium.features.PolyLine([(aPickLat, aPickLong),(aDelLat,aDelLong)],color = 'red', weight = 10.0).add_to(m)
        folium.features.PolyLine([(bPickLat, bPickLong),(bDelLat,bDelLong)],color = 'red', weight = 10.0).add_to(m)

    

    fn='tmp.html'
    tmpurl='file://{path}/{mapfile}'.format(path=os.getcwd(),mapfile=fn)
    m.save(fn)

    browser = webdriver.Chrome(webdriverDir)
    browser.get(tmpurl)
    #Give the map tiles some time to load
    time.sleep(delay)
    browser.save_screenshot('map_{}.png')
    browser.quit()