#!/usr/bin/env python
import matplotlib.pyplot as plt 
from obspy.core import UTCDateTime
import glob 
import numpy as np
import matplotlib as mpl
mpl.rc('font',family='serif')
mpl.rc('font',serif='Times')
mpl.rc('text', usetex=True)
mpl.rc('font',size=18)
def get_values(filehand):
    f = open(filehand,'r')
    coes1, coes2, coes3, res, times = [],[],[], [],[]
    for line in f:
        try:
            line = line.split(',')
            line[0] = line[0].replace(']','').replace('[','')
            coest = line[0].split(' ')
            coes1.append(float(coest[0])) 
            coes2.append(float(coest[1])) 
            coes3.append(float(coest[2]))
            res.append(float(line[1]))
            times.append(UTCDateTime(line[2]))
        except:
            continue

    f.close()
    timesg = []
    for time in times:
        timesg.append(time.year + time.julday/365.25 + time.hour/(365.25*60))
    return coes1, coes2, coes3, res, times, timesg

files = glob.glob('*.csv')
files.sort()

stas, v1, v2, v3, std1, std2, std3 = [],[],[],[],[],[],[]
for cfile in files:
    coes1, coes2, coes3, res, times, timesg = get_values(cfile)
    v1.append(np.mean(coes1))
    v2.append(np.mean(coes2))
    v3.append(np.mean(coes3))
    std1.append(np.std(coes1))
    std2.append(np.std(coes2))
    std3.append(np.std(coes3))
    stas.append(cfile.split('_')[1])




fig = plt.figure(1, figsize=(20,20))
plt.subplot(1,3,1)
for idx, allv in enumerate(zip(v1,v2,v3,std1,std2,std3)):
    plt.subplot(1,3,1)
    plt.plot([allv[0]-allv[3], allv[0]+allv[3]],[idx, idx],color='C0')
    plt.subplot(1,3,2)
    plt.plot([allv[1]-allv[4], allv[1]+allv[4]],[idx, idx],color='C0')
    plt.subplot(1,3,3)
    plt.plot([allv[2]-allv[5], allv[2]+allv[5]],[idx, idx],color='C0')
#plt.subplot(2,1,2)
#for t, re  in zip(timesg, res):
#    plt.plot(t, re/10**9,marker='.')
lets =['(a)','(b)','(c)']
for idx in range(3):
    plt.subplot(1,3,idx+1)
    plt.yticks(range(len(stas)), stas, fontsize=6)
    plt.xlim((-0.3,0.3))
    plt.ylim((0, len(stas)+1))
    plt.text(-0.3, len(stas)+2, lets[idx])
plt.subplot(1,3,1)
plt.ylabel('Station (Code)')



plt.subplot(1,3,2)
plt.xlabel('Sensitivity ($m/s^2/T$)')
plt.savefig('test.png', format='PNG', dpi=400)
plt.savefig('test.png',format='PDF', dpi=400)
