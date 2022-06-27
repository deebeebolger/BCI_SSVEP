
import mne
import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import psd_welch
from scipy import signal
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt


dataIn = sio.loadmat("/Users/bolger/PycharmProjects/BCI_SSVEP/REC/17_41_24_01_2022_freq_15.mat")

## Need to detect the start of each of the trials and create an ndarray of trial data (from X)

trial_onsets  = dataIn["trial"]
data_cont     = dataIn["X"]
trial_freq    = dataIn["Y"]
tt_stamps     = dataIn["trial_time_stamps"]
t_stamps      = dataIn["time_stamps"]

ionset  = np.where(trial_freq > 0)
ioffset = np.where(trial_freq == 0)



tstim_onset  = trial_onsets[:, ionset[1]]
tstim_offset = trial_onsets[:, ioffset[1]]


#####
DataIn = np.squeeze(dataIn["X"])
DataIn2 = DataIn.T
t = dataIn["time_vect"]

ch_names = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
ch_types = ['eeg', 'eeg'] * 4
sampling_freq = 250
info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
print(info)

rawIn = mne.io.RawArray(DataIn2, info, first_samp=0)

t_min = 0
t_max = 117
fmax = 25
fmin = 5
psds, freq = mne.time_frequency.psd_welch(rawIn, fmin=fmin, fmax=fmax, tmin=t_min, tmax=t_max, n_fft=np.shape(t)[1], n_overlap=0,
                             n_per_seg=1500, window='hamming', verbose=False)

fig, axes = plt.subplots(2, 1, sharex='all', sharey='none', figsize=(8, 5))

psds_plot = 10 * np.log10(psds)
axes[0].plot(freq, psds_plot[7,:], color='b')


