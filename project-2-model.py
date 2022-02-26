# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 18:00:30 2022

@author: james
"""

import os, h5py, numpy
from datetime import datetime
from numpy import zeros, arange, nan, real, imag, conj, pi, sin, arctan, arctan2, angle, exp, sqrt, diff, pad, std
from utilities.gpsl1ca import get_GPS_L1CA_code_sequence, acquire_GPS_L1CA_signal, \
    L1CA_CODE_RATE, L1CA_CODE_LENGTH, L1CA_CARRIER_FREQ
from utilities.file_source import get_file_source_info, SampleLoader
from utilities.gpst import dt2gpst, gpst2dt, gpst_week_seconds
from utilities.hdf5_utils import write_dict_to_hdf5
from utilities.hdf5_utils import read_hdf5_into_dict
from utilities.coordinates import geo2ecf




# other libraries 
import math
from scipy.io import loadmat
import scipy.stats as stats
import matplotlib.pyplot as plt

#%%  Choose PRN and tracking parameters,
prn = 5
offset = 0
recordFig = True    # deciding whether to save these figures or not

#%% Loading the data and preparing the workspace
figuresDir_all = './figures/'
fileName = 'G{0:02}_models'.format(prn) 
figuresDir = figuresDir_all+ fileName + '/'

if (recordFig): 
    try:    
        os.mkdir(figuresDir)
        print("Saving Figures in directory: " + figuresDir)
    except:
        recordFig = False
        print("The figure directory for this combination already exists. This will re-create the figures in your environment," 
              " but will not save them, as they likely already exist in the folder.")


#%% Loading the signal model
# loading signal model
signalModel_filepath = './Data/haleakala_20210611_160000_RX7_signal_model.mat'
signalModel_in = loadmat(signalModel_filepath, squeeze_me=True)
signalModel_keys = ['az_dlos', 'az_sp', 'doppler_D', 'doppler_R', 'el_dlos',
                    'el_sp', 'gpsweek', 'sp_lat', 'sp_lon', 'sp_mss', 'tau_D',
                    'tau_R', 'timeVec']
signalModel = {}  #{key: [0] for key in signalModel_keys}
for keys in signalModel_keys:
    signalModel[keys] = signalModel_in[keys]
del signalModel_in

#%% Loading in the navigation data to produce models:
navData_filepath = './Data/haleakala_20210611_160000_RX7_nav.mat'
navData_in = loadmat(navData_filepath, squeeze_me=True)
navData_keys = ['__header__', '__version__', '__globals__', 'Rx_Clk_Bias', 
                'Rx_Clk_Drift', 'Rx_TimeStamp', 'Rx_Vx', 'Rx_Vy', 'Rx_Vz', 
                'Rx_X', 'Rx_Y', 'Rx_Z', 'Rx_height', 'Rx_lat', 'Rx_lon']
navData = {}  #{key: [0] for key in signalModel_keys}
for keys in navData_keys:
    navData[keys] = navData_in[keys]
del navData_in

#%% Producing our own model data

# Useful constants
c = 2.998e8 # m/s

# Satellite and reciever information
t_rx = navData['Rx_TimeStamp']
S_lat,S_lon,S_alt = signalModel['sp_lat'][offset:,prn-1], signalModel['sp_lon'][offset:,prn-1], signalModel['sp_mss'][offset:,prn-1]
Rx, Ry, Rz = navData['Rx_X'], navData['Rx_Y'], navData['Rx_Z']

# Doing linear fits on the receiver location values so they're not as noisy:
# Starting with Rx:
x_lin = numpy.array([stats.linregress(t_rx,Rx).slope,stats.linregress(t_rx,Rx).intercept])
Rx_lin = x_lin[0]*t_rx+x_lin[1]
y_lin = numpy.array([stats.linregress(t_rx,Ry).slope,stats.linregress(t_rx,Ry).intercept])
Ry_lin = y_lin[0]*t_rx+y_lin[1]
z_lin = numpy.array([stats.linregress(t_rx,Rz).slope,stats.linregress(t_rx,Rz).intercept])
Rz_lin = z_lin[0]*t_rx+z_lin[1]

Vx, Vy, Vz = navData['Rx_Vx'], navData['Rx_Vy'], navData['Rx_Vz']
ClockDrift = navData['Rx_Clk_Drift']

# Linear fitting the drift because the values are kinda strange:
d_lin = numpy.array([stats.linregress(t_rx,ClockDrift).slope,stats.linregress(t_rx,ClockDrift).intercept])
CD_lin = d_lin[0]*t_rx+d_lin[1]
ClockBias = navData['Rx_Clk_Bias']

# Converting/calculating values we need:
S_geo = geo2ecf(numpy.array([S_lat, S_lon, S_alt]))
Sx, Sy, Sz = S_geo[:,0],S_geo[:,1], S_geo[:,2]
GeoRange = numpy.sqrt((Sx - Rx_lin)**2 + (Sy - Ry_lin)**2 + (Sz - Rz_lin)**2)
Range = GeoRange - (ClockBias * c) - CD_lin

# Finally calculating the models:
tau_C = (t_rx - (abs(GeoRange)/c) + signalModel['timeVec'])
doppler_C = (c) / (c - Range[1:] / numpy.diff(t_rx) ) # this isn't right, but shows some proportionality
# doppler_C = (c) / (c - numpy.diff(Range) / numpy.diff(t_rx) ) # this is closer to right, but needs some filtering

#%% Plotting the models/difference:
fig = plt.figure(figsize=(10, 6), dpi=200)
axes = [fig.add_subplot(2, 1, 1 + i) for i in range(2)]
ax1, ax2 = axes

ax1.scatter(t_rx[100:-1] - t_rx[0], doppler_C[100:] , s=1, color='b', label='Generated doppler')
# ax1.scatter(t_rx - t_rx[0], Range, s=1, color='b', label='Generated doppler')

ax2.scatter(t_rx - t_rx[0],signalModel['doppler_D'][offset:,prn-1], s=1, color='r', label='Given doppler')

ylim = max(numpy.abs(ax1.get_ylim()))
for ax in axes:
    ax.grid()
    # ax.set_ylim(-1815,-1800)
ax1.set_ylabel('Doppler [Hz]')
ax2.set_ylabel('Doppler [Hz]')
ax2.set_xlabel('Time [seconds]')
ax1.set_xticklabels([])
ax1.legend(markerscale=10, loc=2, framealpha=1)
txt_label = 'Generated and Given Doppler Shifts for G{0:02}'.format(
    prn)
ax1.set_title(txt_label)

if(recordFig):
    figName = figuresDir + 'corrVal_' + fileName + '.png'
    plt.savefig(figName)

plt.show()

#%% Prepping for the challenge question by loading in the data file:
DTU18_filepath = './Data/dtu18.mat'
DTU18 = loadmat(DTU18_filepath, squeeze_me=True)['dtu18']
DTU18_lon = DTU18['lon']
DTU18_lat = DTU18['lat']
DTU18_mss = DTU18['mss']