import mne
import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import psd_welch
from scipy import signal
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
from mne.time_frequency import psd_welch

dataIn = sio.loadmat("/Users/bolger/PycharmProjects/BCI_SSVEP/REC/session_freq5_14_51_24_01_2022.mat")

D = dataIn["X"]
t = dataIn["time_vect"]
tdata = dataIn["trial_data"]
tdata1 = np.squeeze(tdata)
tdata1 = tdata1.T

plt.plot(tdata1[0,:])

## Filter the data
fs = 250
lowcut = 2
highcut = 40
order = 6

nyq = 0.5 * fs
low = lowcut / nyq
high = highcut / nyq
b, a = signal.butter(order, [low, high], btype='band')
datafilt = signal.filtfilt(b, a, tdata1)

plt.plot(tdata1[4,:])
plt.plot(datafilt[4,:])

freqs1, Pxx = signal.welch(tdata1[7,:], 250, nperseg=None)
plt.semilogy(freqs1, Pxx.T)

freqs, Pxx_den = signal.welch(datafilt[7,:].T, 250, nperseg=1000)
plt.semilogy(freqs, Pxx_den.T)

## Create the reference signals for CCA Enhancement
freqsoi = [10, 12, 15, 17, 20]       #the target frrequencies for reference signals
n_harmonics = 1                         #Number of harmonics to take into account
cca = CCA(max_iter = 1000, n_components=1)

targets = {}
#t_vec = np.linspace(0, 15, 29460)
for freq in freqsoi:
    sig_sin, sig_cos = [], []
    for harmonics in range(n_harmonics):
        sig_sin.append(np.sin(2 * np.pi * 1 * freq * np.squeeze(t)))
        sig_cos.append(np.cos(2 * np.pi * 1 * freq * np.squeeze(t)))
    targets[freq] = np.array(sig_sin + sig_cos).T

scores = []
for freqIdx in range(0, np.shape(freqsoi)[0]):

    X_out = cca.fit(datafilt.T, targets[freqsoi[freqIdx]]).transform(datafilt.T)
    sig_c, t_c = cca.fit_transform(datafilt.T, targets[freqsoi[freqIdx]])
    scores.append(np.corrcoef(sig_c.T, t_c.T)[0, 1])


    f1, Pxx_den = signal.welch(X_out.T, 250, nperseg=1024)
    f2, Pxx_stim = signal.welch(datafilt[7,:].T, 250, nperseg=1024)

    plt.subplot(2,3,freqIdx+1)
    plt.semilogy(f1, Pxx_den.T)
    plt.semilogy(f2, Pxx_stim)

print("Most highly correlated to: " + str(freqsoi[np.argmax(scores)]) + "Hz")















