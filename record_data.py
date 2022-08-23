import os

VERBOSE = False
#
# if os.name == "nt":
#     # DIRTY workaround from stackoverflow
#     # when using scipy, a keyboard interrup will kill python
#     # so nothing after catching the keyboard interrupt will
#     # be executed
#
#     import imp
#     import ctypes
#     import thread
#     import win32api
#
#     basepath = imp.find_module('numpy')[1]
#     ctypes.CDLL(os.path.join(basepath, 'core', 'libmmd.dll'))
#     ctypes.CDLL(os.path.join(basepath, 'core', 'libifcoremd.dll'))
#
#     def handler(dwCtrlType, hook_sigint=thread.interrupt_main):
#         if dwCtrlType == 0:
#             hook_sigint()
#             return 1
#         return 0
#
#     win32api.SetConsoleCtrlHandler(handler, 1)


import threading           # NOQA
import scipy.io as sio     # NOQA
import pylsl               # NOQA
import time                # NOQA
import numpy as np

import matplotlib.pyplot as plt
from mne.time_frequency import psd_welch
from scipy import signal
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
from mne.time_frequency import psd_welch


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

    def freqdetect(self, dataIn):

        t = self.time_vect
        N = 250 * 15
        t = t[0:N]
        D = np.transpose(dataIn)
        Dcurr_len = np.shape(D)
        print(f"The length of Dcurr is {Dcurr_len} ")
        print(f"The size of t variable is {np.shape(t)}")
        print(f"{t}")


        fs = 250

        lowcut = 2
        highcut = 40
        order = 6

        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        datafilt = signal.filtfilt(b, a, D)
        print(f"Size of datafilt is {np.shape(datafilt)}")

        ## Create the reference signals for CCA Enhancement
        freqsoi = [10, 12, 15, 17, 20]  # the target frrequencies for reference signals
        n_harmonics = 1  # Number of harmonics to take into account
        cca = CCA(max_iter=1000, n_components=1)

        t = np.arange(0, 3750, 1)*(1/250)
        targets = {}
        # t_vec = np.linspace(0, 15, 29460)
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
