import os, numpy, h5py

from utilities.gpsl1ca import L1CA_CODE_RATE, L1CA_CODE_LENGTH, L1CA_CARRIER_FREQ
from utilities.hdf5_utils import read_hdf5_into_dict

import matplotlib as mpl
import matplotlib.pyplot as plt

from numpy import pi

#%% Loading the data and preparing the workspace
output_dir = './tracking-output/'
filenames = sorted(os.listdir(output_dir))
outputs = []
for filename in filenames:
    filepath = os.path.join(output_dir, filename)
    with h5py.File(filepath, 'r') as f:
        outputs.append(read_hdf5_into_dict(f))

print('\n'.join(['{0: >2}: {1}'.format(i, fn) for i, fn in enumerate(filenames)]))

#%% Preparing the workspace
i = 0               # this is the file number (arbitrary) out of all output files
recordFig = True    # deciding whether to save these figures or not

figuresDir_all = './figures/'
fileName = 'G{0:02}_{1:02}ms_OLR_Q2'.format(
    outputs[i]['prn'], 1000 * outputs[i]['integration_time']) ## I changed the output file name depending on the task
figuresDir = figuresDir_all+ fileName + '/'

if (recordFig): 
    try:    
        os.mkdir(figuresDir)
        print("Saving Figures in directory: " + figuresDir)
    except:
        recordFig = False
        print("The figure directory for this combination already exists. This will re-create the figures in your environment," 
              " but will not save them, as they likely already exist in the folder.")

# useful values
totTime= outputs[i]['time'][-1]
minTime, maxTime = 0,60
# e, p, l = outputs[i]['e_idx'],outputs[i]['i_idx'], outputs[i]['l_idx']
e,p,l = 0,1,2
#%% Plotting the early, late, and prompt correlator results across I and Q
# (navigation bits are INCLUDED)
fig = plt.figure(figsize=(10, 6), dpi=200)
axes = [fig.add_subplot(2, 1, 1 + i) for i in range(2)]
ax1, ax2 = axes

ax1.scatter(outputs[i]['time'], outputs[i]['prompt'].real, s=1, color='b', label='Prompt')
ax1.scatter(outputs[i]['time'], outputs[i]['early'].real, s=1, color='y', label='Early')
ax1.scatter(outputs[i]['time'], outputs[i]['late'].real, s=1, color='r', label='Late')

ax2.scatter(outputs[i]['time'], outputs[i]['late'].imag, s=1, color='r')
ax2.scatter(outputs[i]['time'], outputs[i]['early'].imag, s=1, color='y')
ax2.scatter(outputs[i]['time'], outputs[i]['prompt'].imag, s=1, color='b')

ylim = max(numpy.abs(ax1.get_ylim()))
for ax in axes:
    ax.grid()
    ax.set_ylim(-ylim, ylim)
    ax.set_xlim(minTime,maxTime)
ax1.set_ylabel('I')
ax2.set_ylabel('Q')
ax2.set_xlabel('Time [seconds]')
ax1.set_xticklabels([])
ax1.legend(markerscale=10, loc=2, framealpha=1)
txt_label = 'Correlation Values Across IQ channels\n G{0:02}: {1:02} ms, Open Loop, with navigation bits'.format(
    outputs[i]['prn'], 1000 * outputs[i]['integration_time'])
ax1.set_title(txt_label)

if(recordFig):
    figName = figuresDir + 'corrVal_' + fileName + '.png'
    plt.savefig(figName)

plt.show()
#%% Plotting EPL correlators, nav. bits removed

# loading the nav bits
navBits= numpy.array([outputs[i]['nav_bits_e'], outputs[i]['nav_bits_p'], outputs[i]['nav_bits_l']])
navBits_avg = numpy.around(numpy.mean(navBits/2+0.5, 0)) * 2 - 1
E_noBits = outputs[i]['early'] * navBits[p]
P_noBits = outputs[i]['prompt'] * navBits[p] 
L_noBits = outputs[i]['late'] * navBits[p]

# E_noBits = outputs[i]['early'] * navBits_avg
# P_noBits = outputs[i]['prompt'] * navBits_avg
# L_noBits = outputs[i]['late'] * navBits_avg

# plotting
fig = plt.figure(figsize=(10, 6), dpi=200)
axes = [fig.add_subplot(2, 1, 1 + i) for i in range(2)]
ax1, ax2 = axes

ax1.scatter(outputs[i]['time'], P_noBits.real, s=1, color='b', label='Prompt')
ax1.scatter(outputs[i]['time'], E_noBits.real, s=1, color='y', label='Early')
ax1.scatter(outputs[i]['time'], L_noBits.real, s=1, color='r', label='Late')

ax2.scatter(outputs[i]['time'], P_noBits.imag, s=1, color='b', label='Prompt')
ax2.scatter(outputs[i]['time'], E_noBits.imag, s=1, color='y', label='Early')
ax2.scatter(outputs[i]['time'], L_noBits.imag, s=1, color='r', label='Late')

ylim = max(numpy.abs(ax1.get_ylim()))
for ax in axes:
    ax.grid()
    # ax.set_ylim(-ylim, ylim)
    ax.set_xlim(minTime,maxTime)
ax1.set_ylabel('I')
ax2.set_ylabel('Q')
ax2.set_xlabel('Time [seconds]')
ax1.set_xticklabels([])
ax1.legend(markerscale=10, loc=2, framealpha=1)
txt_label = 'Correlation Values Across IQ channels\n G{0:02}: {1:02} ms, Open Loop, with navigation bits REMOVED'.format(
    outputs[i]['prn'], 1000 * outputs[i]['integration_time'])
ax1.set_title(txt_label)

if(recordFig):
    figName = figuresDir + 'corrVal_noBits_' + fileName + '.png'
    plt.savefig(figName)

plt.show()

#%% Plotting correlation magnitude
fig = plt.figure(figsize=(10, 6), dpi=200)
ax = fig.add_subplot(111)

ax.scatter(outputs[i]['time'], abs(outputs[i]['prompt']), s=1, color='b', label='Prompt')
ax.scatter(outputs[i]['time'], abs(outputs[i]['early']), s=1, color='y', label='Early')
ax.scatter(outputs[i]['time'], abs(outputs[i]['late']), s=1, color='r', label='Late')

ylim = max(numpy.abs(ax.get_ylim()))
ax.grid()
ax.set_ylim(0, ylim)
ax.set_xlim(0, 60)
ax.set_ylabel('IQ Magnitude')
ax.set_xlabel('Time [seconds]')
ax.set_xticklabels([])
ax.legend(markerscale=10, loc=2, framealpha=1)
txt_label = 'Correlation Magnitudes \n G{0:02}: {1:02} ms, Open Loop'.format(
    outputs[i]['prn'], 1000 * outputs[i]['integration_time'])
ax.set_title(txt_label)

if(recordFig):
    figName = figuresDir + 'corrMag_' + fileName + '.png'
    plt.savefig(figName)
    
plt.show()

#%% Plotting Detrended code and carrier phase

# Compute code and carrier phase trends
elapsed_time = outputs[i]['time'] - outputs[i]['time'][0]
ave_doppler = numpy.mean(outputs[i]['doppler'])

adjusted_code_rate = L1CA_CODE_RATE * (1 + ave_doppler / L1CA_CARRIER_FREQ)
code_phase_trend = elapsed_time * adjusted_code_rate

residual_carr_phase = outputs[i]['carr_phase'] - elapsed_time * ave_doppler
carr_phase_trend = elapsed_time * ave_doppler + numpy.polyval(numpy.polyfit(elapsed_time, residual_carr_phase, 2), elapsed_time)

# Making the plots
fig = plt.figure(figsize=(10, 6), dpi=200)
axes = [fig.add_subplot(2, 1, 1 + i) for i in range(2)]
ax1, ax2 = axes

ax1.scatter(outputs[i]['time'], outputs[i]['code_phase'] - code_phase_trend, s=1, color='b', label='Measured')
ax1.set_ylabel('Detr. Code Phase [chips]')

ax2.scatter(outputs[i]['time'], outputs[i]['carr_phase'] - carr_phase_trend, s=1, color='b', label='Measured')
ax2.set_ylabel('Detr. Carrier Phase [cycles]')

for ax in axes:
    ax.grid()
    ax.set_xlim(0, maxTime)
ax1.legend(markerscale=10, loc=2, framealpha=1)
ax1.set_xticklabels([])
ax2.set_xlabel('Time [seconds]')
fig.align_labels()

txt_label = 'Code and Carrier Phase Over Time \n G{0:02}: {1:02} ms, Open Loop'.format(
    outputs[i]['prn'], 1000 * outputs[i]['integration_time'])
ax1.set_title(txt_label)

if(recordFig):
    figName = figuresDir + 'phase_' + fileName + '.png'
    plt.savefig(figName)

plt.show()

#%% Computing CNR 
def compute_cnr(I, Q, bandwidth):
    noise_var = numpy.var(Q)
    return (I**2 + Q**2) / noise_var * bandwidth

#%% Plotting CNR
fig = plt.figure(figsize=(10, 3), dpi=200)
ax = fig.add_subplot(111)

indices = [i]
for j, i in enumerate(indices):
    color = plt.cm.viridis(j / len(indices))
    cnr = compute_cnr(outputs[i]['prompt'].real, outputs[i]['prompt'].imag, 1 / outputs[i]['integration_time'])
    ax.scatter(outputs[i]['time'], 10 * numpy.log10(cnr), s=1, color=color, label='Measured')

ax.set_ylabel('C/N0 [dB]')
ax.set_xlabel('Time [seconds]')
ax.grid()
ylim = (numpy.abs(ax.get_ylim()))
ax.set_xlim(0, maxTime)
ax.set_ylim(ylim[0], ylim[1])
txt_label = 'C/N0 Over Time \n G{0:02}: {1:02} ms, Open Loop'.format(
    outputs[i]['prn'], 1000 * outputs[i]['integration_time'])
ax.set_title(txt_label)

if(recordFig):
    figName = figuresDir + 'CNR_' + fileName + '.png'
    plt.savefig(figName)

plt.show()

#%% Plotting discriminator for 2 or 4 quadrants as indicated above:
phase_q2 = numpy.unwrap(outputs[i]['disc_q2'] *2) / 2 / (2 * pi) * 360
phase_q4 = numpy.unwrap(outputs[i]['disc_q4'] *2) / 2 / (2 * pi) * 360
    
fig = plt.figure(figsize=(10, 6), dpi=200)
axes = [fig.add_subplot(2, 1, 1 + i) for i in range(2)]
ax1, ax2 = axes

ax1.scatter(outputs[i]['time'], phase_q2, s=1, color='b', label='2-Quad. Disc')
ax2.scatter(outputs[i]['time'], phase_q4, s=1, color='r', label='4-Quad. Disc')

ylim = max(numpy.abs(ax1.get_ylim()))
for ax in axes:
    ax.grid()
    # ax.set_ylim(-ylim, ylim)
    ax.set_xlim(minTime,maxTime)
ax1.set_ylabel('Angle [deg]')
ax2.set_ylabel('Angle [deg]')
ax2.set_xlabel('Time [seconds]')
ax1.set_xticklabels([])
ax1.legend(markerscale=10, loc=2, framealpha=1)
txt_label = 'Comparing 2 and 4 quadrent discriminators \n G{0:02}: {1:02} ms, Open Loop'.format(
    outputs[i]['prn'], 1000 * outputs[i]['integration_time'])
ax1.set_title(txt_label)

if(recordFig):
    figName = figuresDir + 'discrims_' + fileName + '.png'
    plt.savefig(figName)

plt.show()