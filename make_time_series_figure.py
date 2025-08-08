#!/usr/bin/env python
from obspy.core import read 
import glob 



#!/usr/bin/env python
import matplotlib.pyplot as plt
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime
import numpy as np 
import matplotlib as mpl
from obspy.geodetics import locations2degrees



mpl.rc('font',family='serif')
#mpl.rc('font',serif='Times')
#mpl.rc('text', usetex=True)
mpl.rc('font',size=18)

#Get seismometer and magnetometer data from IRIS
#Find stations sensitive to geomagnetic storms
client = Client("IRIS")

#Timescale = current solar cycle

stime = UTCDateTime('2024-10-10T00:00:15')
etime = UTCDateTime('2025-06-01T00:00:00')
#etime = stime + 2*24*60*60

#net, sta, loc, chan = "AK", "DOT", "*", "LHZ"

sRMS, mRMS = [], []

def grabData(sncl):
    net, sta, loc, chan = sncl.split(",")
    mnet, msta, mloc, mchan = "IU", "COLA", "40", "*FZ,*F1,*F2"
    ctime = stime

    ltime = []

    #Grab data in 3 hour increments and analyze
    inv = client.get_stations(network=net, starttime=stime, 
            endtime = etime, level="response", 
        location=loc, channel=chan, station=sta)


    mnet, msta, mloc, mchan = find_geomagsta(inv)
    print('HERE IS THE STATION:' + msta)



    while ctime < etime:
        print(ctime)
        if True:
        #try:
            #st = read_rover_data(net, sta, loc, chan, ctime)
            
            st = client.get_waveforms(net, sta, loc, chan, ctime-10*60, ctime+3*60*60 + 10*60)
            
            mst = client.get_waveforms(mnet, msta, mloc, mchan, ctime, ctime+3*60*60)
            #mst = read_mag_data(mnet, msta, mloc, mchan, ctime)
            #mst = read_rover_data(mnet, msta, mloc, mchan, ctime)
            st.trim(ctime-10*60, ctime+3*60*60 + 10*60)
            mst.trim(ctime-10*60, ctime+3*60*60 + 10*60)
        # except:
        #     print('Bad data for ' + sta + ' ' + str(ctime))
        #     ctime += 3*60*60
        #     continue




        st.detrend("constant")
        mst.detrend("constant")
        try:
            st.merge(fill_value = 0.)
            mst.merge(fill_value = 0.)
        except:
            ctime += 3*60*60
            continue
        try:
            st.remove_response(inventory=inv, output='ACC')
            
            #mst[0].data /= 41.9430 # nT
            st.filter('bandpass', freqmin=1/500., freqmax=1/200.)
            mst.filter('bandpass', freqmin=1/500., freqmax=1/200.)
            st.trim(ctime, ctime+3*60*60)
            mst.trim(ctime, ctime+3*60*60)
        except:
            ctime += 3*60*60
            continue

        if len(mst) < 1:
            ctime += 3*60*60
            continue

        if mst[0].stats.sampling_rate > 1:
            mst.decimate(5)
            mst.decimate(2)
        try:
            sRMS.append(st[0].std())
        except:
            ctime += 3*60*60
            continue


       

        return st, mst




def read_rover_data(net, sta, loc, chan, stime):
    string_read = '/research/rover_archive/rover_AK/data/' + net
    string_read += '/' + str(stime.year) + '/' + str(stime.julday).zfill(3) 
    string_read += '/' + sta + '.' + net + '.' + str(stime.year) + '.' + str(stime.julday).zfill(3)

    st = read(string_read)
    if stime.hour == 0:
        ctime = stime -24*60*60
        string_read = '/research/rover_archive/rover_AK/data/' + net
        string_read += '/' + str(ctime.year) + '/' + str(ctime.julday).zfill(3) 
        string_read += '/' + sta + '.' + net + '.' + str(ctime.year) + '.' + str(ctime.julday).zfill(3)
        st += read(string_read)
    if stime.hour >= 21:
        ctime = stime + 24*60*60
        string_read = '/research/rover_archive/rover_AK/data/' + net
        string_read += '/' + str(ctime.year) + '/' + str(ctime.julday).zfill(3) 
        string_read += '/' + sta + '.' + net + '.' + str(ctime.year) + '.' + str(ctime.julday).zfill(3)
        st+= read(string_read)

    return st


def read_mag_data(net, sta, loc, chan, stime):
    string = '/msd/IU_COLA'
    string += '/' + str(stime.year) + '/' + str(stime.julday).zfill(3) + '/*40_*F*.seed'
    st = read(string)
    if stime.hour == 0:
        ctime = stime - 24*60*60
        string = '/msd/IU_COLA'
        string += '/' + str(ctime.year) + '/' + str(ctime.julday).zfill(3) + '/*40_*F*.seed'
        st += read(string)
    if stime.hour >= 21:
        ctime = stime +24*60*60
        string = '/msd/IU_COLA'
        string += '/' + str(ctime.year) + '/' + str(ctime.julday+1).zfill(3) + '/*40_*F*.seed'
        st += read(string)

    return st
 

def grab_sncls(net_code, channelCode):

    inv = client.get_stations(network=net_code, station='*',
        starttime=stime, endtime=etime, channel=channelCode, location='*', level = "channel")
    print(inv)
    sncl = []
    for net in inv:
        for sta in net:
            for chan in sta:
                sncl.append(net_code + "," + str(sta.code) +"," + chan.location_code + "," + str(chan.code))
    return sncl


def find_geomagsta(inv):
    # Find the geomag station most near the station

    minv = client.get_stations(network='NT', starttime=stime, 
            endtime = etime)

    lat = inv[0][0].latitude
    lon = inv[0][0].longitude
    cdis = 10000.
    for net in minv:
        for sta in net:
            if sta.code in ['BRT', 'HOT', 'GUT','FDT', 'CMT']:
                continue
            mlat = sta.latitude
            mlon = sta.longitude
            dist = locations2degrees(lat, lon, mlat, mlon)
            if dist < cdis:
                mnet, msta, mloc, mchan = 'NT', sta.code, 'R0', 'LFF'
                cdis = dist

    return mnet, msta, mloc, mchan


#sncls = [net +"," + sta +"," +loc+ "," +chan, "IU,COLA,00,LHZ"]

if __name__ == '__main__':
    from multiprocessing import Pool
    #sncls = grab_sncls("AK", "LHZ")
    sncls = ['AK,A21K,*,LHZ', 'AK,N18K,*,LHZ']

    





    print(sncls)
    #p = Pool(40)
    #p.map(grabData, sncls)
    fig = plt.figure(1,figsize=(12,12))

    for idx, sncl in enumerate(sncls):
        st, mst = grabData(sncl)


        plt.subplot(2,1,1)
        plt.plot(st[0].times()/60., st[0].data*10**9, label=(st[0].id).replace('.',' '))
        plt.ylabel('Acceleration (nm/s/s)')
        plt.legend(loc='upper right')
        if idx == 1:
            plt.text( -25, plt.ylim()[1], '(a)')
        plt.xlim((0, 180))
        plt.subplot(2,1,2)
        plt.plot(mst[0].times()/60., mst[0].data, label=(mst[0].id).replace('.',' ')) 
        plt.xlabel('Time (minutes)')
        if idx == 1:
            plt.text( -25, plt.ylim()[1], '(b)')
        plt.ylabel('Field Strength (nT)')
        plt.legend(loc='upper right')
        plt.xlim((0, 180))
    plt.savefig('tsmag.png')
    plt.savefig('tsmag.pdf')



# fig, ax = plt.subplots(1, 1, figsize=(16,16))
# plt.plot(sRMS, mRMS, ".")
# plt.xlabel("Seismic Root Mean Squares")
# plt.ylabel("Magnetic Root Mean Squares")
# plt.savefig('Record_Section_body.PDF', format='PDF', dpi=400)
# plt.savefig('Record_Section_body.PNG', format='PNG', dpi=400)

#To Do
#Remove response into filtering