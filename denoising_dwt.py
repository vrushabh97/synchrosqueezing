import pywt
import speechpy
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
fs, data = wavfile.read('H:/dataset/msjs1/msjs1/audio/sa1.wav')
w = pywt.Wavelet('sym4')
data_dwt_max_val = pywt.dwt_max_level(len(data),w)
print(data_dwt_max_val)
coeffs = pywt.wavedec(data,w,level = 2)
Ca2,Cd2,Cd1 = coeffs
Ca2t = pywt.threshold(Ca2,np.std(Ca2)/2,mode= 'soft')
Cd2t = pywt.threshold(Cd2, np.std(Cd2),mode= 'less')
Cd1t = pywt.threshold(Cd1, np.std(Cd1),mode= 'less')
denoised_data = pywt.waverec([Ca2t,Cd2t,Cd1t], w, mode='symmetric', axis=-1)
plt.plot(data)
plt.plot(denoised_data)
plt.show()
wavfile.write('H:/dataset/msjs1/msjs1/audio/xyz.wav',fs,denoised_data/np.max(denoised_data))
