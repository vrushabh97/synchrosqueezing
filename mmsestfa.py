import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.io import wavfile
def segment(signal,W,SP,wnd):
    L = len(signal)
    SP = np.fix(W*SP)
    N = np.fix((L-W)/SP+1)
    A = np.arange(0,N)[np.newaxis]
    Index = np.tile(np.arange(1,W+1),(int(N),1)) + np.tile(A.T*SP,(1,int(W)))
    #Window = wnd[np.newaxis]
    hw = np.tile(wnd,(int(N),1))
    #print(Index.astype(int))
    Seg = signal[Index.astype(int)]*hw
    print(np.shape(Seg),np.shape(hw))
    return Seg

        
fs, data = wavfile.read('H:/dataset/msjs1/msjs1/audio/sa1.wav')
IS = 0.25
W = np.fix(0.025*fs)
SP = 0.4
wnd = np.hamming(W)
"""plt.plot(wnd)
plt.show()"""
pre_emph = 0.0
signal = sig.lfilter(np.array([1.0 -pre_emph]),np.array([1.0]),data)
NIS = np.fix((IS*fs-W)/(SP*W)+1)
y = segment(signal,W,SP,wnd)
Y = np.fft.fft(y)
Yphase = np.angle()
#plt.plot(y[0])
#plt.show()
