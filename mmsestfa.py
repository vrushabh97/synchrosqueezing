import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.io import wavfile
import scipy.special as sp
from math import *
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
    #print(np.shape(Seg),np.shape(hw))
    return Seg
def vad(signal,noise,NoiseCounter):
    NoiseMargin = 3
    Hangover = 8
    FreqResol = len(signal)
    SpectralDist = 20*(np.log10(signal)-np.log10(noise))
    SpectralDist[np.where(SpectralDist<0)] = 0
    Dist = np.mean(SpectralDist)
    if Dist<NoiseMargin:
        NoiseFlag = 1
        NoiseCounter+=1
    else:
        NoiseFlag =0
        NoiseCounter = 0
    if NoiseCounter > Hangover:
        SpeechFlag = 0
    else:
        SpeechFlag = 1
    return NoiseFlag, SpeechFlag, NoiseCounter, Dist   

        
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
Yphase = np.angle(Y[:,np.arange(0,int(np.shape(Y)[1]/2+1))])
Y = np.abs(Y[:,np.arange(0,int(np.shape(Y)[1]/2+1))])
numberofFrames = np.shape(Y)[0]
FreqResol = np.shape(Y)[1]
N = np.mean(Y[0:int(NIS)-1,:],0)
LambdaD = np.mean(np.square(Y[0:int(NIS)-1,:]),0)
alpha = 0.99
NoiseCounter = 0;
NoiseLength = 9
G = np.ones(len(N))
Gamma = G
Gamma1p5 = sp.gamma(1.5)
X = np.zeros(np.shape(Y))
for i in range(0, numberofFrames):
    #print(i)
    if i<=NIS:
        SpeechFlag = 0
        NoiseCounter = 100
    else:
        #print(i)
        NoiseFlag,SpeechFlag,NoiseCounter,Dist = vad(Y[i,:],N,NoiseCounter)
    if SpeechFlag == 0:
        N = (NoiseLength*N+Y[i,:])/(NoiseLength+1)
        LambdaD=(NoiseLength*LambdaD+np.square(Y[i,:]))/(1+NoiseLength)
    gammaNew = (np.square(Y[i,:]))/LambdaD
    xi=alpha*(np.square(G))*Gamma+(1-alpha)*np.max(gammaNew-1,0)
    Gamma = gammaNew
    nu = Gamma*xi/(1+xi)
    G = (Gamma1p5*np.sqrt(nu)/Gamma)*np.exp(-nu/2)*((1+nu)*sp.jv(0,nu/2)+nu*sp.jv(1,nu/2)      
    #np.where(np.isnan(G) or np.isinf(G))
    G[np.where(np.isnan(G) or np.isinf(G))] = xi[np.where(np.isnan(G) or np.isinf(G))]/(1+xi[np.where(np.isnan(G) or np.isinf(G))])
    X[i,:] = G*Y[i,:]
#output = OverlapAdd2(X,Yphase,W,SP*W)
#signal = sig.lfilter(np.array([1.0],np.array([1.0 -pre_emph])),output)                                                    
#plt.plot(y[0])
#plt.show()
