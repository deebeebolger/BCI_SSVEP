import numpy as np
import matplotlib.pyplot as plt

rfresh_rate = 60
f = 10
seq = []

for counter in range(0, rfresh_rate):
    X = 2*np.pi*(counter/rfresh_rate)
    s = 0.5*(1+ np.sin(X))
    seq.append(s)

plt.plot(seq, 'o')
plt.xlabel("Frame index")
plt.ylabel("Luminance")
plt.title("Sampled Sinusoidal Stimulation Sequence (Frame Rate = 60Hz")
plt.show()

