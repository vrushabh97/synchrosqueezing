import pywt
import speechpy
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
fs, data = wavfile.read('H:/dataset/msjs1/msjs1/audio/sa1.wav')
w = pywt.Wavelet('bior6.8')
data_dwt_max_val = pywt.dwt_max_level(len(data),w)
print(data_dwt_max_val)
coeffs = pywt.wavedec(data,w,level = 4)
Ca2,Cd4,Cd3,Cd2,Cd1 = coeffs
#Ca2t = pywt.threshold(Ca2,np.std(Ca2)/2,mode= 'soft')
Cd4t = pywt.threshold(Cd4, 20000,mode= 'hard')
Cd3t = pywt.threshold(Cd3, 20000,mode= 'hard')
Cd2t = pywt.threshold(Cd2, 20000,mode= 'hard')
Cd1t = pywt.threshold(Cd1, 20000,mode= 'hard')
denoised_data = pywt.waverec([Ca2,Cd4t,Cd3t,Cd2t,Cd1t], w, mode='symmetric', axis=-1)
plt.plot(data)
plt.plot(denoised_data)
plt.show()
wavfile.write('H:/dataset/msjs1/msjs1/audio/xyz.wav',fs,denoised_data/np.max(denoised_data))
