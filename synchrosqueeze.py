import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
gaussian_window = signal.gaussian(300,std = 10)
fs,audio = wavfile.read('H:/dataset/msjs1/msjs1/audio/sa1.wav')
f, t, Zxx = signal.stft(audio, fs,nperseg = 1024,nfft = 1024,noverlap = 1000)

fig, ax = plt.subplots()
ax.set_yscale('linear')
ax.pcolormesh(t, f, 20*np.log(abs(Zxx)))
plt.show()
