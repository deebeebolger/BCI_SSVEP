# StreamSSVEP
# Streams continuous, raw EEG data via LSL.
# This script has been written specifically to stream data from openBCI cyton board (8-channels).
# Note:
# Sampling frequency = 250Hz
# Number of scalp channels = 8
# Scale factor for EEG data is a multiplier that converts the EEG values from counts into volts.
# The estimation of the scale factor for EEG data here is based on a maximum gain of 24; this is the default
# gain set by the Arduino sketch that is running on the OpenBCI board.
# Note that you need to define the correct port address ('/dev/cu.usbserial-DM03H6G8' for mac)
# ************************************************************************************************************

from pylsl import StreamInfo, StreamOutlet, local_clock
import numpy as np

#gain = 24
#SCALE_FACTOR_EEG = (4500000)/gain/(2**23-1) #uV/count
#SCALE_FACTOR_AUX = 0.002 / (2**4)


print("Creating LSL stream for EEG. \nName: OpenBCIEEG\nID: EmotivtestEEG\n")

info_eeg = StreamInfo('EmotivEEG', 'EEG', 8, 128, 'float64', 'OpenBCItestEEG')

#print("Creating LSL stream for AUX. \nName: OpenBCIAUX\nID: OpenBCItestEEG\n")

#info_aux = StreamInfo('OpenBCIAUX', 'AUX', 3, 250, 'float32', 'OpenBCItestAUX')

outlet_eeg = StreamOutlet(info_eeg)

def lsl_streamers(sample):
	outlet_eeg.push_sample(np.array(sample.channels_data)*SCALE_FACTOR_EEG)
	outlet_aux.push_sample(np.array(sample.aux_data)*SCALE_FACTOR_AUX)

board = OpenBCICyton(port='/dev/cu.usbserial-DM03GSQ0')   #DM03H5W5
board.start_stream(lsl_streamers)