import pylsl

channel_data = []
time_stamps = []

streams = pylsl.resolve_stream('type', 'EEG')
inlet = pylsl.stream_inlet(streams[0])

while True:

        sample, time_stamp = inlet.pull_sample()
        time_stamp += inlet.time_correction()

        print(time_stamp)
        print(sample)
        time_stamps.append(time_stamp)
        channel_data.append(sample)



