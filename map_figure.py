
import pygmt
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime




# TO DO List
# Make map fill piece of paper





# We want to make some code that produces a map of Alaska with stations
# We will grab the stations and metadata from EarthScope

debug = True
client = Client("IRIS")
font = "10p,Helvetica"
marker, t ='c0.3c', 't0.3c'
stime = UTCDateTime('2019-12-01T00:00:00')
etime = UTCDateTime('2025-06-09T00:00:00')


# Here is where we make the actual map

grid = pygmt.datasets.load_earth_relief(resolution="10m")
fig = pygmt.Figure()
fig.basemap(projection="T200/28c", region=[170, 240, 50, 75], frame="a10f10g10")
fig.grdimage(grid=grid, projection="T200/28c", cmap="globe")
fig.coast(borders="1/0.1p,black")


# Function to grab lats and lons
def grab_lats_lons(net_code, channelCode):

    inv = client.get_stations(network=net_code, station='*',
        starttime=stime, endtime=etime, channel=channelCode, location='*')
    if debug:
        print(inv)
    lats, lons, stas= [], [],[]
    for net in inv:
        for sta in net:
            lats.append(sta.latitude)
            lons.append(sta.longitude)
            stas.append(sta.code)
    return lats, lons, stas



lats, lons, stas = grab_lats_lons('AK', 'LHZ')

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
nlats = [54, 57, 60, 63, 66, 69]
nlons = [-170, -160, -150, -140, -130]
gl= ax.gridlines(draw_labels=True, xlocs=nlons, ylocs=nlats)
gl.xlabels_top = False
gl.ylabels_left = False
import matplotlib.ticker as mticker
gl.xlocator = mticker.FixedLocator(nlons)
gl.ylocator = mticker.FixedLocator(nlats)
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

plt.tight_layout()

sc = ax.scatter(x = lons, y = lats, s = 100, marker = "^", edgecolors = 'black', c="red", label= 'Seismometers', transform=ccrs.Geodetic())

mg = ax.scatter(x = mlons, y = mlats, s = 100, edgecolors = 'black', c="blue", label= 'Magnetometers', transform=ccrs.Geodetic())
ax.legend()

fig.savefig('Alaska_Station_Map' + '.pdf')
fig.savefig('Alaska_Station_Map' + '.jpg')