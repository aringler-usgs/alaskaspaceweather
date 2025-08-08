import glob
import csv
import pandas as pd
import pygmt
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime

# files = glob.glob("./newdata/new_data/images/*.pdf")
f = pd.read_table("./newdata/stations.csv", header = None)
client = Client("IRIS")

# count = 0

# # for file in files:
# #     count+=1

lats, lons, stas = [], [], []
nlats, nlons, nstas = [], [], []

print(f)

print(f[0])

for item in f[0]:
    if(item.strip().startswith("#")):
        if(len(item) == 56):
            inv = client.get_stations(network = "AK", channel = "LHZ", station = item[30:33])
        else:
            inv = client.get_stations(network = "AK", channel = "LHZ", station = item[30:34])
        for net in inv:
            for sta in net:
                lats.append(sta.latitude)
                lons.append(sta.longitude)
                stas.append(sta.code)
    else:
        if(len(item) == 55):  
            inv = client.get_stations(network = "AK", channel = "LHZ", station = item[29:32])
        else:
            inv = client.get_stations(network = "AK", channel = "LHZ", station = item[29:33])
        for net in inv:
            for sta in net:
                nlats.append(sta.latitude)
                nlons.append(sta.longitude)
                nstas.append(sta.code)
        # inv = client.get_stations(network = "AK", channel = "LHZ", station = item[30:33])
    
print(nstas)
print(stas)

print(len(nstas))
print(len(stas))

def grab_lats_lons(net_code, channelCode):

    inv = client.get_stations(network=net_code, station='*', channel=channelCode, location='*')
    print(inv)
    lats, lons, stas= [], [],[]
    for net in inv:
        for sta in net:
            lats.append(sta.latitude)
            lons.append(sta.longitude)
            stas.append(sta.code)
    return lats, lons, stas

mlats, mlons, mstas = grab_lats_lons('NT', '*')

"""
Map making
"""


#Make map with res_var values as color coded!
import csv
import pygmt
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime
import pandas as pd
from obspy.core import UTCDateTime, read
from obspy.core.inventory import read_inventory
import glob
import sys

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from cartopy.feature import NaturalEarthFeature

mpl.rc('font',family='serif')
mpl.rc('font',serif='Times')
#mpl.rc('text', usetex=True)
mpl.rc('font',size=20)

states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor=cfeature.COLORS['land'])

def setupmap(central_lon, central_lat,handle):
    #handle = plt.axes(projection=ccrs.AlbersEqualArea(central_lon, central_lat))
    handle.set_extent(extent)

    handle.add_feature(cfeature.LAND.with_scale('50m'), edgecolor='gray',facecolor=cfeature.COLORS['land'])
    handle.add_feature(cfeature.LAND)
    handle.add_feature(cfeature.OCEAN.with_scale('50m'),facecolor=cfeature.COLORS['water'], edgecolor='gray'  )
    handle.add_feature(states_provinces, edgecolor='gray')
    handle.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor='gray',facecolor=cfeature.COLORS['land'] )
    handle.add_feature(cfeature.LAKES, alpha=0.5)
    handle.add_feature(cfeature.RIVERS)

    return handle

boxcoords=[min(lats) -1., min(lons)-1., max(lats) +1. , max(lons) + 1.]
extent=[boxcoords[1], boxcoords[3], boxcoords[0], boxcoords[2]]
central_lon = np.mean(extent[:2])
central_lat = np.mean(extent[2:])   

fig= plt.figure(figsize=(14,14))
ax = plt.subplot(1,1,1, projection=ccrs.EquidistantConic(central_lon, central_lat))
ax = setupmap(central_lon, central_lat, ax)

fig.canvas.draw()
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
hlats = [54, 57, 60, 63, 66, 69]
hlons = [-170, -160, -150, -140, -130]
gl= ax.gridlines(draw_labels=True, xlocs=nlons, ylocs=nlats)
gl.xlabels_top = False
gl.ylabels_left = False
import matplotlib.ticker as mticker
gl.xlocator = mticker.FixedLocator(hlons)
gl.ylocator = mticker.FixedLocator(hlats)
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

plt.tight_layout()

sc = ax.scatter(x = lons, y = lats, s = 100, marker = "^", edgecolors = 'black', c="steelblue", label= 'Seismometers', transform=ccrs.Geodetic())
mg = ax.scatter(x = nlons, y = nlats, s = 100, marker = "^", edgecolors = 'black', c="red", label= 'Seismometers with Non-Linear Fit', transform=ccrs.Geodetic())
hg = ax.scatter(x = mlons, y = mlats, s = 70, edgecolors = 'black', c="cyan", label= 'Magnetometers', transform=ccrs.Geodetic())

ax.legend()

fig.savefig('linearmaps' + '.pdf')
fig.savefig('linearmaps' + '.jpg')



