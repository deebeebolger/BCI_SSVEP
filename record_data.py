
import os
import threading
import scipy
import scipy.io as sio
import pylsl
import time
import numpy as np
import warnings

from scipy import signal
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt

VERBOSE = False

def time_str():
    return time.strftime("%H_%M_%d_%m_%Y", time.gmtime())


class NoRecordingDataError(Exception):
    def __init__(self):
        self.value = "Received no data while recording"

    def __str__(self):
        return repr(self.value)


class KillSwitch():
    def __init__(self):
        self.terminate = False

    def kill(self):
        self.terminate = True
        return False


def record(channel_data=[], time_stamps=[], KillSwitch=None, time_vect=[], data_noinc=[]):
    if VERBOSE:
        sio.savemat("recording_" + time_str() + ".mat", {
            "time_stamps"  : [1, 2, 3],
            "channel_data" : [1, 2, 3]
        })
    else:
        print('Hello there Dee!')
        streams = pylsl.resolve_stream('type', 'EEG')
        inlet = pylsl.stream_inlet(streams[0])
        sampcount = 0

        while True:
            try:
                sample, time_stamp = inlet.pull_sample()
                time_stamp += inlet.time_correction()
                curr_samp = sampcount*(1/250)

                time_stamps.append(time_stamp)
                channel_data.append(sample)   # Vector with a column for each channel
                data_noinc.append(sample)

                time_vect.append(curr_samp)
                sampcount += 1

                # print("Current time: "   + str(time_stamp))
                # print("Current sample: " + str(sample))
                # print(" Channel 1: "     + str(sample[0]))
                # print(" Channel 2: "     + str(sample[1]))
                # print(" Channel 3: "     + str(sample[2]))

                # first col of one row of the record_data matrix is time_stamp,
                # the following cols are the sampled channels

            except KeyboardInterrupt:
                complete_samples = min(len(time_stamps), len(channel_data))
                sio.savemat("recording_" + time_str() + ".mat", {
                    "time_stamps"  : time_stamps[:complete_samples],
                    "channel_data" : channel_data[:complete_samples]
                })
                break
    if KillSwitch.terminate:
        return False

def cca_reference(list_freqs, fs, num_smpls, num_harms):

    num_freqs = len(list_freqs)
    tidx = np.arange(1, num_smpls + 1) / fs  # time index

    y_ref = np.zeros((num_freqs, 2 * num_harms, num_smpls))
    for freq_i in range(num_freqs):
        tmp = []
        for harm_i in range(1, num_harms + 1):
            stim_freq = list_freqs[freq_i]  # in HZ
            # Sin and Cos
            tmp.extend([np.sin(2 * np.pi * tidx * harm_i * stim_freq),
                        np.cos(2 * np.pi * tidx * harm_i * stim_freq)])
        y_ref[freq_i] = tmp  # 2*num_harms because include both sin and cos

    return y_ref


def filterbank(eeg, fs, idx_fb):

    """
        Adapted from https://github.com/eugeneALU/CECNL_RealTimeBCI/blob/master/filterbank.py
        Created on Fri Nov 1 2019
        Author eugeneALU
    """
    if idx_fb == None:
        warnings.warn('stats:filterbank:MissingInput ' \
                      + 'Missing filter index. Default value (idx_fb = 0) will be used.')
        idx_fb = 0
    elif (idx_fb < 0 or 9 < idx_fb):
        raise ValueError('stats:filterbank:InvalidInput ' \
                         + 'The number of sub-bands must be 0 <= idx_fb <= 9.')

    if (len(eeg.shape) == 2):
        num_chans = eeg.shape[0]
        num_trials = 1
    else:
        num_chans, _, num_trials = eeg.shape

    # Nyquist Frequency = Fs/2N
    Nq = fs / 2

    passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
    stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
    Wp = [passband[idx_fb] / Nq, 90 / Nq]
    Ws = [stopband[idx_fb] / Nq, 100 / Nq]
    [N, Wn] = signal.cheb1ord(Wp, Ws, 3, 40)  # band pass filter StopBand=[Ws(1)~Ws(2)] PassBand=[Wp(1)~Wp(2)]
    [B, A] =  signal.cheby1(N, 0.5, Wn, 'bandpass')  # Wn passband edge frequency

    y = np.zeros(eeg.shape)
    if (num_trials == 1):
        for ch_i in range(num_chans):
            # apply filter, zero phass filtering by applying a linear filter twice, once forward and once backwards.
            # to match matlab result we need to change padding length
            y[ch_i, :] = signal.filtfilt(B, A, eeg[ch_i, :], padtype='odd', padlen=3 * (max(len(B), len(A)) - 1))

    else:
        for trial_i in range(num_trials):
            for ch_i in range(num_chans):
                y[ch_i, :, trial_i] = signal.filtfilt(B, A, eeg[ch_i, :, trial_i], padtype='odd',
                                                            padlen=3 * (max(len(B), len(A)) - 1))

    return y

def plotFreqDetect(refreq, rho, Edata, srate):


    # Filter the current trial EEG data before plotting spectrum.
    lowcut = 2
    highcut = 30
    order = 8
    nyq = 0.5 * srate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    datafilt = signal.filtfilt(b, a, Edata.T)
    print(f"Size of datafilt is {np.shape(datafilt)}")

    datafilt_mean = np.mean(Edata, 1)
    fig, axs = plt.subplots(1, 2)

    # Plot bar graph of weighted sum of squares of correlation values.
    axs[0].set_title("Weighted sum of squares (WSS) of correlations ")
    axs[0].bar(refreq, rho, color='blue')
    axs[0].set_ylabel("WSS of correlations")
    axs[0].set_xlabel("Reference frequency (Hz)")

    # Plot the log magnitude spectrum of the input data.
    # Calculate the welch estimate
    Fxx, Pxx = scipy.signal.welch(datafilt_mean, fs=srate, window='hanning')
    idx = np.argmin(np.abs(Fxx - 30))
    axs[1].set_title("PSD (Welch method) of current-trial EEG data")
    axs[1].plot(Fxx[0:idx], 10*np.log(Pxx[0:idx]))
    axs[1].set_xlabel("Frequency (Hz")
    axs[1].set_ylabel("Magnitude (PSD)")

    fig.tight_layout()
    plt.show()


class RecordData():
    def __init__(self, Fs, age, gender="male", record_func=record):
        # timepoints when the subject starts imagination
        self.trial = []

        self.X = []

        self.trial_time_stamps = []
        self.time_stamps       = []
        self.time_vect         = []
        self.trial_timevect    = []
        self.trial_data        = []
        self.X_noninc          = []

        self.killswitch = KillSwitch()
        # containts the lables of the trials:
        # TODO add frequency label mapping
        # 1:
        # 2:
        # 3:
        # 4:
        self.Y = []

        # sampling frequncy
        self.Fs = Fs

        self.gender   = gender
        self.age      = age
        self.add_info = ""

        recording_thread = threading.Thread(group=None,
            target=record_func,
            args=(self.X, self.time_stamps, self.killswitch, self.time_vect, self.X_noninc),
        )
        recording_thread.daemon = True
        self.recording_thread   = recording_thread

    def __iter__(self):
        yield 'trial'            , self.trial
        yield 'age'              , self.age
        yield 'X'                , self.X
        yield 'time_stamps'      , self.time_stamps
        yield 'trial_time_stamps', self.trial_time_stamps
        yield 'Y'                , self.Y
        yield 'Fs'               , self.Fs
        yield 'gender'           , self.gender
        yield 'add_info'         , self.add_info
        yield 'time_vect'        , self.time_vect
        yield 'trial_data'       , self.trial_data
        yield 'trial_timevect'   , self.trial_timevect

    def add_trial(self, label, to_add=[]):

        if label == 0:
            N = 250 * 15  # Define the 15second interval
            curr_data = self.X_noninc
            print("Trial length in samples is:" + str(len(curr_data[-N:])))
            self.trial_data.append(curr_data[-N:])  # Pass only the 15seconds to the frequency analysis.
            to_add = curr_data[-N:]
            print("current trial data dimension: " + str(np.shape(curr_data[-N:])))
            print("Pause Label: " + str(label))

        self.trial_time_stamps.append(pylsl.local_clock())  # Get the time at the start of the trial.
        if label == 0:
            self.trial_data.append(to_add)
        self.Y.append(label)
        self.trial_timevect.append(self.time_vect[-1])

        if label > 0:
            print("Trial Label: " + str(label))

        return to_add

    def freqdetect(self, dataIn, forig):

        """
        Method to calculate the SSVEP using standard CCA.
        :param dataIn:
        :return:
        """

        D = np.transpose(dataIn)
        Dcurr_len = np.shape(D)
        fs = 250
        N = fs * 15
        t = np.arange(0, 3750, 1) * (1 / fs)

        print(f"The length of Dcurr is {Dcurr_len} ")
        print(f"The size of t variable is {np.shape(t)}")
        print(f"{t}")

        lowcut = 2
        highcut = 40
        order = 6

        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        datafilt = signal.filtfilt(b, a, D)
        print(f"Size of datafilt is {np.shape(datafilt)}")

        # Instantiate the CCA object.
        cca = CCA(max_iter=1000, n_components=2)

        # Generate the sine-cosine based reference signals. Here we define 1 harmonic component
        ## Initial the frequency of the reference signals
        freqsoi = [8, 10, 12, 15, 17, 20]  # the target frrequencies for reference signals
        n_harmonics = 1  # Number of harmonics to take into account

        targets = {}
        for freq in freqsoi:
            sig_sin, sig_cos = [], []
            for harmonics in range(n_harmonics):
                sig_sin.append(np.sin(2 * np.pi * 1 * freq * np.squeeze(t)))
                sig_cos.append(np.cos(2 * np.pi * 1 * freq * np.squeeze(t)))
            print(f"the shape of sig_sin is {np.shape(sig_sin)}\n ")
            print(f"the shape of sig_cos is {np.shape(sig_cos)}\n ")
            x = np.array(sig_sin + sig_cos)
            targets[freq] = np.array(sig_sin + sig_cos).T
            print(f"the size of targets is {targets}")

        scores = []
        for freqIdx in range(0, np.shape(freqsoi)[0]):
            #X_out = cca.fit(datafilt.T, targets[freqsoi[freqIdx]]).transform(datafilt.T)
            sig_c, t_c = cca.fit_transform(datafilt.T, targets[freqsoi[freqIdx]])
            scores.append(np.corrcoef(sig_c.T, t_c.T)[0, 1])

        print("Most highly correlated to: " + str(freqsoi[np.argmax(scores)]) + "Hz")

    def freqdetect_fbcca(self, dataIn, forig):

        """
        Detection of SSVEPs using filter-bank canonical correlation analysis (fbcca).
        :param dataIn: current trial data for all n channels
        It prints out the frequency that most closely matches the stimulation frequency according to fbcca analysis.
        If the maximum  correlation value is too low, it will not give a result
        """
        numfbs = 5  # The number of filterbanks
        n_harms = 2  # Number of harmonics
        fs = 250  # Sampling frequency

        freqlist = np.arange(8,24,1)
        num_samps, num_chans = np.shape(dataIn)      # The imported data has shape time-points X channel
        num_targets = len(freqlist)
        print(f"the dimension of input data is {np.shape(dataIn)}\n")
        dataIn = np.array(dataIn)
        print(print(f"the dimension of input data is {dataIn.shape}\n"))

        # Define the filterbank coefficients to weight the sub-band components.
        # Based on finding that SSVEP SNR decreases as a function of increasing response frequency.
        # Constants -1.25 and 0.25 maximize classification performance.
        fb_coef = np.power(np.arange(1, numfbs+1),(-1.25)) + 0.25

        # Generate the reference sine-cosine based signals
        sigref = cca_reference(freqlist, fs, num_samps, n_harms)  # Call of function cca_reference

        # Instantiate CCA object
        cca = CCA(max_iter=1000, n_components=2)

        # # Initialize a results matrix
        # result matrix
        res = np.zeros(( numfbs, num_targets))

        for ifb in range(numfbs):
            testd = filterbank(dataIn.T, fs, ifb)

            for iclass in range(num_targets):
                 dataref = np.squeeze(sigref[iclass, :, :])
                 print(print(f"the dimension of input dataref is {dataref.shape}\n"))
                 test_C, ref_C = cca.fit_transform(testd.T, dataref.T)
                 print(f"The type of test_c is {type(test_C)} and type of ref_C is {type(ref_C)}\n")
                 print(f"The size of test_c is {test_C.shape} and type of ref_C is {ref_C.shape}\n")
                 temp_res, _ = pearsonr(test_C.flatten(), ref_C.flatten())
                 print(type(temp_res))
                 if temp_res == np.nan:
                    temp_res = 0
                 res[ifb, iclass] = temp_res

        print(f"size of res is {np.shape(res)}")

        # # Calculate the weighted sum of r from all the different filter banks results
        sum_r = np.dot(fb_coef, res)
        print(f"The output weighted sum of correlations all filterbanks is {sum_r}\n")


        # Get the maximum from the target as the final predict. It returns the index.
        finalres_i = np.argmax(sum_r)
        # '''Set the threshold correlation'''
        Threshold = 2.1
        if abs(sum_r[finalres_i])< Threshold:
             print(f"Original stimulus frequency is: {forig}\n.")
             print(f"The correlation of {sum_r[finalres_i]} is too low. \n")
             print(f"Frequency {freqlist[finalres_i]}Hz is most likely with a correlation of {sum_r[finalres_i]}\n")
        else:
             print(f"Original stimulus frequency is: {forig}\n.")
             print(f"Frequency {freqlist[finalres_i]}Hz is most likely with a correlation of {sum_r[finalres_i]}\n")

        plotFreqDetect(freqlist, sum_r, dataIn, fs)   # Call of function to plot the frequency spectrum and fbCCA results



    def start_recording(self):
        self.recording_thread.start()
        time.sleep(16)
        if len(self.X) == 0:
            raise NoRecordingDataError()

    def set_trial_start_indexes(self):
        i = 0
        for trial_time_stamp in self.trial_time_stamps:
            for j in range(i, len(self.time_stamps)):
                time_stamp = self.time_stamps[j]
                if trial_time_stamp <= time_stamp:
                    self.trial.append(j - 1)
                    i = j
                    break

    def stop_recording_and_dump(self, file_name="session_" + time_str() + ".mat"):
        self.set_trial_start_indexes()
        sio.savemat(file_name, dict(self))

        return file_name


if __name__ == '__main__':
    record()
