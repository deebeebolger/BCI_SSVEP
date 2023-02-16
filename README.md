# BCI_SSVEP
# Brain Computer Interface SSVEP

A very simple stand-alone implementation of an SSVEP protocol in the python language that is based on the acquisition of EEG data using the OpenBCI system. 
EEG signals are directly transferred from the Cyton board of the OpenBCI system using the pyOpenBCI library. 
The most recent version (January 2023) applies a **filter-bank Canonical Correlation Analysis (filterbank CCA)** on each trial to carry out the frequency detection. 

# Notes for using the suite of functions:

*Note firstly that the following is susceptible to change in further versions of the program.*

Open up a first terminal and run “StreamSSVEP.py” to create an LSL stream for the EEG data. 
The data is streamed directly from the Cyton board via Lab Streaming Layer (LSL) protocol using the pylsl, which is the Python interface of the LSL. This is made possible using the “OpenBCICyton()” function of the pyOpenBCI library. 

The data samples are scaled to microV using the following scaling factor:  4.5 Volts/Gain/(223 -1)
We need to scale the data as the data streamed from the Cyton board is the raw data in counts read by the board. 
Note that each board has a specific scale factor. The gain is 25x.

Note: Initializing the board differs between platforms

**For Windows replace '*' with the port number
board = OpenBCICyton(port='COM*')**

**For MacOS and Linux replace '*' with the port number
board = OpenBCICyton(port='/dev/ttyUSB*')**

If you don’t know which COM port you’re connected to, you can either use the OpenBCI GUI to inform you or assign “None” to the port number 
and use the find_port() function. This function will connect to the first Cyton Dongle that it detects. 

## Running the SSVEP Program:

Once the stream from the Cyton board has been established, we can open a second terminal to launch the SSVEP program proper.
The user has two possibilities:

1.  **python3 RunSSVEP.py exp1 randomize 15** 

Run this so that the flickering frequency is randomized. 
The first argument, exp1 (arg[1]), defines the protocol to run. For the moment exp1 is the protocol that works.
The second argument defines the frequency of the flickering checkerboard. Here, the frequency of the flickering randomly varies from one trial to the next and the possible frequencies is defined in the “begin_experiment_1()” function. 
The third argument defines the duration (in seconds) of the flickering stimulus; here the duration is 15seconds. 

At the moment, the number of trials to run is defined upon calling the “begin_experiment_1()” function as follows:

- def begin_experiment_1(freq,duration, trials=5)**

The definition of the number of trials could be set in the terminal as an extra argument and accessed as arg[4]…

2. **python3 RunSSVEP.py exp2 15**

Here a specific flickering frequency (arg[2]) is defined (e.g. 15). 
This is useful for testing the success rate of the script in detecting the equivalent frequency in the EEG spectrum. 


*To do: To define the flickering frequencies to be randomized, we could use the refresh rate (in Hertz) of the screen used in the SSVEP protocol.* 
*The refresh rate of the screen can be determined using the “get_fps”:*

**clock = pygame.time.Clock()
fps = int(clock.get_fps())**

*A possible problem is that this code will not return a precise fps (e.g. 61Hz or 59Hz), so it may help to round to the nearest 10…this will need to be tested.* 

## References


Chen X, Wang Y, Gao S, Jung TP, Gao X. Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based brain-computer interface. J Neural Eng. 2015 Aug;12(4):046008. doi: 10.1088/1741-2560/12/4/046008. Epub 2015 Jun 2. PMID: 26035476.

Rabiul Islam M, Khademul Islam Molla M, Nakanishi M, Tanaka T. Unsupervised frequency-recognition method of SSVEPs using a filter bank implementation of binary subband CCA. J Neural Eng. 2017 Apr;14(2):026007. doi: 10.1088/1741-2552/aa5847. Epub 2017 Jan 10. PMID: 28071599.

https://xribenesite.wordpress.com/2017/11/22/ssvep-bci-arduino-based-configurable-leds-hardware-design/

https://xribenesite.wordpress.com/2017/11/24/ssvep-bci-protocol-design-and-offline-processing/

https://github.com/HeosSacer/SSVEP-Brain-Computer-Interface

Other sources :
https://github.com/aaravindravi/Brain-computer-interfaces

















