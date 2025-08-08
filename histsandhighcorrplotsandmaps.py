import pandas as pd
import matplotlib.pyplot as plt
import math
import glob
from scipy.optimize import curve_fit
import numpy as np
import sys
import os
import shutil
import scipy.odr as odr

import matplotlib as mpl
from obspy.core import UTCDateTime
import csv
import statistics

mpl.rc('font',family='serif')
mpl.rc('font',serif='Times')
mpl.rc('text')
mpl.rc('font',size=18)


def calc_bin_percentile(mag,seismic, percent, bins, minm, maxm):
    
    perts, binvg = [], []
    seismict = seismic[(mag <= maxm) & (mag >= minm)]
    magt = mag[(mag <= maxm) & (mag >= minm)]
    binvals = np.logspace(np.log10(minm), np.log10(maxm), bins)
    print(binvals)
    print(binvals)
    for idx, binv in enumerate(binvals[:-1]):
        data = seismict[(magt > binv) & (magt <= binvals[idx+1])]
        if len(data) > 0:
            perts.append(np.percentile(data, percent))
            binvg.append(binv)

    return binvg, perts


# Following scipy ODR code see docs.scipy.org/doc/scipy/reference/odr.html
def linearFunction(B, x):
    return B[0]*x + B[1]

def readData(fileName):
    seismic = pd.read_csv(fileName, usecols=[0])
    seismic = np.array(seismic).flatten()
    seismic *= 10**9
    mag = pd.read_csv(fileName, usecols = [1])
    mag = np.array(mag).flatten()
    # fig, ax = plt.subplots(1, 1, figsize=(16,16))
    # plt.plot(mag,seismic,'.', alpha=0.5, color='C1', label='All Data')
    times = pd.read_csv(fileName, usecols =[2])
    times = np.array(times).flatten()
    corrs = pd.read_csv(fileName, usecols=[4])
    corrs = np.array(corrs).flatten()

    allcounts = mag

    #mag = 20*np.log10(mag)
    #seismic = 20*np.log10(seismic)
    # We should clean the data
    mag = mag[((corrs)>= 0.7)]
    seismic = seismic[((corrs)>= 0.7)]
    timesg = times[(corrs>=0.7)]
    # mag = mag[(seismic <= np.median(seismic) + 1.*np.std(seismic))]
    # seismic = seismic[(seismic <= np.median(seismic) + 1.*np.std(seismic))]
    # seismic = seismic[(mag <= np.median(mag) + 1.*np.std(mag))]
    # mag = mag[(mag <= (np.median(mag) + 1.*np.std(mag)))]
    #plt.plot(mag,'b')

    print(np.median(mag) + 1.*np.std(mag))

    binvals, perts = calc_bin_percentile(mag, seismic, 5, 10, 0.1, 40)
    # plt.loglog(binvals, perts, label='Percent')
    # Straight from scipy.odr
    linear = odr.Model(linearFunction)
    mydata = odr.Data(binvals, perts, wd = 1., we=1.)
    myodr = odr.ODR(mydata, linear, beta0=[1.,1.])
    myoutput = myodr.run()
    print(myoutput.pprint())

    newtimes = []
    for ctime in timesg:
        ctime = UTCDateTime(ctime)
        newtimes.append(ctime.year + ctime.julday/365.25 + ctime.hour/(365.25*24))

    
    #redplots

    """
    plt.loglog(mag, seismic, '.', label='Correlation$\geq$0.7', color='C3')
    #plt.loglog(seismic, mag, ".")
    plt.ylabel("Seismic Data (nm/s/s)")
    plt.xlabel("Magnetic Data (nT)")
    xdata = np.linspace(min(binvals), max(binvals), 100)
    ydata = linearFunction(myoutput.beta, xdata)
    plt.loglog(xdata, ydata, 'C5-', label = "Curve Slope:" + str(myoutput.beta[0]) + ' m/s/s/T')
    plt.legend()
    pdfName = fileName.replace('.csv','.pdf')
    pngName = fileName.replace('.csv','.png')
    # plt.savefig(pdfName, format='PDF', dpi=400)
    # plt.savefig(pngName, format='PNG', dpi=400)
    # plt.clf()
    # plt.close()
    plt.show()
    """

    corrs = mag
    slope = myoutput.beta[0]
    ratio = len(corrs)/len(allcounts)

    return corrs, allcounts, slope, ratio


def makeHist(fileName):
    corrs, allcounts, slope, ratio = readData(fileName)
    print("High Corrs: ")
    print(corrs)
    print("All mag data: ")
    print(allcounts)
    fig = plt.figure(1, figsize = (12,12))
    plt.subplot(2,1,1)
    plt.hist(allcounts, bins = 100, range = (0, 20))
    plt.ylabel("# All Counts")
    plt.xlabel("Magnetic Field")
    # plt.xlim(0, 2500)
    plt.subplot(2,1,2)
    plt.ylabel("# High Correlation Counts")
    plt.xlabel("Magnetic Field")
    # plt.xlim(0, 2000)
    plt.hist(corrs, bins = 100, range = (0, 20))
    plt.savefig(fileName.replace(".csv", "") + 'histogram' + '.pdf')
    plt.savefig(fileName.replace(".csv", "") + 'histogram' + '.jpg') 
    plt.clf()
    plt.close()      

files = glob.glob("./newdata/new_data/*.csv")
for file in files:
    try:
        makeHist(file)
    except:
        print("Bad Data from:" + file)


def getSlopesandRatios(files):
    stas = []
    slopes = []
    ratios = []
    bad = []
    for fileName in files:
        try:
            corrs, allcounts, slope, ratio = readData(fileName)
            slopes.append(slope)
            ratios.append(ratio)
            stas.append(fileName.replace("./newdata/new_data/AK_", "").replace("__LHZ_rover_GEO_TRY.csv", "").replace("_*_LHZ_rover_GEO_TRY.csv", ""))
        except:
            print(fileName)
            bad.append(fileName)

    print(stas)
    print(slopes)
    print(len(stas))
    print(len(slopes))
    print(bad)
    return stas, slopes, ratios



# Map making


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

lats, lons = [], []

# grid = pygmt.datasets.load_earth_relief(resolution="10m")
fig = pygmt.Figure()
# fig.basemap(projection="T200/28c", region=[170, 240, 50, 75], frame="a10f10g10")
# # fig.grdimage(grid=grid, projection="T200/28c", cmap="globe")
# fig.coast(borders="1/0.1p,black")
marker, t ='c0.3c', 't0.3c'

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


client = Client("IRIS")

ss = []

files = glob.glob("./newdata/new_data/AK*.csv")
stas, slopes, ratios = getSlopesandRatios(files)
for index, sta in enumerate(stas):
    try:
        inv = client.get_stations(network = "AK", station = sta, channel = "LHZ")
        for net in inv:
            for s in net:
                lats.append(s.latitude)
                lons.append(s.longitude)
                ss.append(s.code)
    except:
        continue

indices = ["COLD", "MESA", "TRF", "FALS"]

for sta in indices:
    try:
        index = ss.index(sta)
        print(index)
        ss.pop(index)
        lats.pop(index)
        lons.pop(index)
        if (sta == "FALS"):
            slopes.pop(index)
            ratios.pop(index)
    except:
        print("not in file")

boxcoords=[min(lats) -1., min(lons)-1., max(lats) +1. , max(lons) + 1.]
extent=[boxcoords[1], boxcoords[3], boxcoords[0], boxcoords[2]]
central_lon = np.mean(extent[:2])
central_lat = np.mean(extent[2:])

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

fig= plt.figure(figsize=(14,14))
ax = plt.subplot(1,1,1, projection=ccrs.EquidistantConic(central_lon, central_lat))
ax = setupmap(central_lon, central_lat, ax)

sc = ax.scatter(x = lons, y = lats, s = 100, edgecolors = 'black', marker = '^', c="red", label= 'Sensitive Seismometers', vmin = 0, transform=ccrs.Geodetic())
mg = ax.scatter(x = mlons, y = mlats, s = 70, edgecolors = 'black', c="blue", label= 'Magnetometers', transform=ccrs.Geodetic())

plt.colorbar(sc, shrink = 0.5, pad = 0.1)

fig.canvas.draw()
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
lats = [54, 57, 60, 63, 66, 69]
lons = [-170, -160, -150, -140, -130]
gl= ax.gridlines(draw_labels=True, xlocs=lons, ylocs=lats)
gl.xlabels_top = False
gl.ylabels_left = False
import matplotlib.ticker as mticker
gl.xlocator = mticker.FixedLocator(lons)
gl.ylocator = mticker.FixedLocator(lats)
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

plt.tight_layout()

plt.savefig('ratiomap' + '.pdf')
plt.savefig('ratiomap' + '.jpg')

#PiYG

plt.show()

"""

#Quantifying 
#mean, std, median? of slopes, ratio, 

smean = 0
sstd = 0
rmean = 0
rstd = 0
corrmean

info = {
    "slope_mean": statistics.mean(slopes),
    "slope_std": statistics.stdev(slopes),
    "ratio_mean": statistics.mean(ratios),
    "ratio_std": statistics.stdev(ratios)
}

print(info)

"""

# import glob
# import os
# import shutil

# pngfiles = glob.glob("*.png")
# pdffiles = glob.glob("*.pdf")

# destination = "./images"

# for file_path in pngfiles:
#     dst_path = os.path.join(destination, os.path.basename(file_path))
#     shutil.move(file_path, dst_path)

# for file_path in pdffiles:
#     dst_path = os.path.join(destination, os.path.basename(file_path))
#     shutil.move(file_path, dst_path)