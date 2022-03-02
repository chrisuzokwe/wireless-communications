# -*- coding: utf-8 -*-
"""ECET 512 HW7.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wNCm2vvqB48cwTGaWzhD0rhe6z1C_HJx
"""

import numpy as np
import matplotlib.pyplot as plt
import random

"""# Fading Statistics

#### Generating Fading Signals
"""

def rayleighfade(N, fm):
  # return rayleigh time samples

  df = (2*fm)/(N-1)
  T = 1/df

  # generating doppler spectrum
  spectrum = np.arange(-fm, fm+(df*1/2), df)

  dspectrum = np.sqrt([1.5/(np.pi*fm*np.sqrt(1-(f/fm)**2)) for f in spectrum])
  dspectrum[0]=2*dspectrum[1]-dspectrum[2]
  dspectrum[-1]=2*dspectrum[-2]-dspectrum[-3] 

  # Gaussian RVs and Frequency Spectrum
  pos = np.array([])
  pos2 = np.array([])

  for i in range(int(N/2)):
    pos = np.append(pos, [np.random.normal(0, 1) + 1j*np.random.normal(0, 1)])
    pos2 = np.append(pos2, [np.random.normal(0, 1) + 1j*np.random.normal(0, 1)])

  neg = np.flip(np.conj(pos))
  neg2 = np.flip(np.conj(pos2))

  gaus1 = np.concatenate((neg, pos))
  gaus2 = np.concatenate((neg2, pos2))

  # Shaping Gaussians by Dspectrum
  X = gaus1*np.sqrt(dspectrum)
  Y = gaus2*np.sqrt(dspectrum)

  tX = np.abs(np.fft.ifft(X))**2
  tY = np.abs(np.fft.ifft(Y))**2

  r = np.sqrt(tX+tY)

  return r, T

r20, T20 = rayleighfade(100, 20)
r20rms = np.sqrt(sum(r20**2)/100)
r20norm = r20/r20rms

r200, T200 = rayleighfade(100, 200)
r200rms = np.sqrt(sum(r200**2)/100)
r200norm = r200/r200rms

plt.figure(figsize=(25, 10))

plt.xlabel('Time (sec)')
plt.ylabel('Magnitude (dB)')

plt.grid(True)
plt.title('Rayleigh Fading Channel: fm=20\n @RMS = 0dB \n AFD  = 0.061875 \n LCR per 1sec=10.101 \n @RMS = -10dB \n AFD = 0.02475 \n LCR per 1sec=2.8282')
plt.xlim(0, 2.5)
t20 =np.arange(0, T20, T20/100)
plt.plot(t20[:100], 20*np.log10(r20norm), color='green', label='Amplitude')
plt.axhline(0, label='RMS LEVEL', color='k', linestyle='--')
plt.axhline(-10, label='10 dB below RMS LEVEL', color='red', linestyle='--')
plt.legend()

plt.figure(figsize=(25, 10))

plt.xlabel('Time (sec)')
plt.ylabel('Magnitude (dB)')

plt.grid(True)
plt.title('Rayleigh Fading Channel: fm=200\n @RMS = 0dB \n AFD  = 0.0058 \n LCR per 1sec=101.01 \n @RMS = -10dB \n AFD = 0.003024 \n LCR per 1sec=36.36')
plt.xlim(0, 0.25)
t200 =np.arange(0, T200, T200/100)
plt.plot(t200[:100], 20*np.log10(r200norm), color='green', label='Amplitude')
plt.axhline(0, label='RMS LEVEL', color='k', linestyle='--')
plt.axhline(-10, label='10 dB below RMS LEVEL', color='red', linestyle='--')
plt.legend()

"""#### Zero-Crossing """

def crossing_rate(sig, thresh):

  above = True if sig[0] > thresh else False
  crossings = 0

  for i in sig:
    if above and i < thresh:
      crossings += 1
      above = False

    elif not above and i > thresh:
      above = True
    
    else:
      pass

  return crossings

# 20fm crossing rate
c20t0 = crossing_rate(20*np.log10(r20norm), 0)
c20t10 = crossing_rate(20*np.log10(r20norm), -10)
print(c20t0)
print(c20t10)

# Level Crossing Rate
lcr20t0 = c20t0*(1/T20)
lcr20t10 = c20t10*(1/T20)
print(lcr20t0)
print(lcr20t10)

# 200fm crossing rate
c200t0 = crossing_rate(20*np.log10(r200norm), 0)
c200t10 = crossing_rate(20*np.log10(r200norm), -10)

print(c200t0)
print(c200t10)

# Level Crossing Rate
lcr200t0 = c200t0*(1/T200)
lcr200t10 = c200t10*(1/T200)
print(lcr200t0)
print(lcr200t10)

"""#### Average Fade Duration"""

def average_fade(sig, time, thresh):
  above = True if sig[0] > thresh else False
  crossings = 0
  fadetotal = 0
  intervals = 0
  dt = 0

  for sample, time in zip(sig, time):

    if above and sample < thresh:
      crossings += 1
      above = False
      dt = time

    elif not above and sample > thresh:
      above = True
      fadetotal += time-dt
      intervals += 1
    
    else:
      pass

  return fadetotal/intervals

t20 =np.arange(0, T20, T20/100)
afd20fm = average_fade(20*np.log10(r20norm), t20, 0)
print(afd20fm)

t20 =np.arange(0, T20, T20/100)
afd20fm10 = average_fade(20*np.log10(r20norm), t20, -10)
print(afd20fm10)

t200 =np.arange(0, T200, T200/100)
afd200fm = average_fade(20*np.log10(r200norm), t200, 0)
print(afd200fm)

t200 =np.arange(0, T200, T200/100)
afd200fm = average_fade(20*np.log10(r200norm), t200, -10)
print(afd200fm)

"""# Bit Error Rate"""

# generate bit string
bits = np.zeros(10000, dtype=int)

for i in range(10000):
  bits[i] = random.randint(0, 1)

# create encoded bit sequence
bitsencoded = []

for i in range(int(len(bits)/2)):

  b1 = bits[2*i]
  b2 = bits[2*i+1]
  
  if b1 and b2:
    bitsencoded.append(-1+1j)
  elif (not b1) and b2:
    bitsencoded.append(1+1j)
  elif b1 and (not b2):
    bitsencoded.append(-1-1j)
  elif (not b1) and (not b2):
    bitsencoded.append(1-1j)

ZdB = np.arange(3,20)
Z = 10**(ZdB/10)

1/Z

z= np.linspace(0, 1, 500)
print(z)

def ispositive(x): return 1 if x >= 0 else 0

BERs = []

for i in z:
  # generate channel h(t) samples
  r, T = rayleighfade(10000, 20)
  theta = np.exp(-1j*np.random.uniform(0, 2*np.pi, 10000))
  h = r*theta
  n = np.random.normal(0, 1, 10000)*i

  # recover signal
  cnoise = n/h
  eq_bits = bitsencoded + cnoise[:5000]

  # decode recieved signal
  eq_decoded = []

  for i in eq_bits:

    real = np.real(i)
    imag = np.imag(i)

    eq_decoded.append(ispositive(real*-1))
    eq_decoded.append(ispositive(imag))

  eq_decoded = np.array(eq_decoded)

  err = 0
  for i in range(len(bits)):

    if bits[i] != eq_decoded[i]:
      err += 1

  BER = err/10000
  BERs.append(BER)

Zdb = 10*np.log10(z)

plt.figure(figsize=(25, 10))

plt.xlabel('Eb/No (dB)')
plt.ylabel('BER')

plt.grid(True)
plt.title('Bit Error Rate vs. SNR')
plt.ylim(10E-4, 5E-1)
plt.plot(-Zdb, BERs, color='red')

"""# BER vs MU"""

import numpy as np
import math
from cmath import exp
from math import cos
from math import sin

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, rc
from IPython.display import HTML
import cmath

ni = 1
nj = 2
N = ni**2 + ni*nj + nj**2
center = [0,0]
radius = 2000/np.sqrt(3)
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
subscripts = ['\u2081', '\u2082', '\u2083', '\u2084', '\u2085', '\u2086', '\u2087']
frequencies = [1.8E9, 1.82E9, 1.84E9, 1.86E9, 1.88E9, 868E6, 882E6]

def hexangles(x): return np.pi*x/6

def drawCell(center,radius,label):
    pts = list(range(0,14,2))
    theta = list(map(hexangles,pts))
    xs = center[0]+radius*np.cos(theta)
    ys = center[1]+radius*np.sin(theta)
    plt.plot(xs,ys,'b-')
    plt.text(center[0],center[1],label)

def findServingCell(mobileLocation, cellCenters):
  distances = [np.sqrt((mobileLocation[0]-cell[0])**2 + (mobileLocation[1]-cell[1])**2)  for cell in cellCenters]
  return distances.index(min(distances)), min(distances)

channels = [[0,0]]
zippedcells = []

fig = plt.figure()
ax = plt.gca()
ax.set_aspect(1)

ax.set_title('Array Antenna Uplink Animation')
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')

plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
numFrames = 60
ims=[]
mobilePosX = np.linspace(-50, 50,numFrames)
mobilePosY = np.linspace(-50,50,numFrames)


mobileAngle = np.arange(0, 2*np.pi, 2*np.pi/numFrames)
#mobilePosX = [40*cos(x) for x in mobileAngle]
#mobilePosY = [40*sin(x) for x in mobileAngle]



zippedcells = []
distances = []
targetcells = []
drawCell([0,0], 100, 'Smart Array')
angles = []

for frames in range(numFrames):
   
   # Find the corresponding serving cell
   idx, distance = findServingCell([mobilePosX[frames],mobilePosY[frames]], [[0,0]])

   #print(channels[idx][0])
   distances.append(distance)
   #targetcells.append(zippedcells[idx])
   # Draw a line connecting the center (basestation) of the serving cell 
   # and the mobile user
   #im, = plt.plot([0,mobilePosX[frames]],[0,mobilePosY[frames]], marker = 'x', color = 'red', animated=True)

   im, = plt.plot( [0,mobilePosX[frames]], [0,mobilePosY[frames]], marker = 'x', color = 'red', animated=True)
   theta = math.atan2(mobilePosY[frames], mobilePosX[frames])
   #theta = (theta+2*np.pi)%(2*np.pi)
   angles.append(theta)



   # Draw the mobile user at the appropriate location
   #im2, = plt.plot(mobilePosX[frames],mobilePosY[frames],'r+', animated=True)
    
   ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

rc('animation', html='jshtml')
ani
#ani.save('user.gif', writer='pillow')
#plt.show()

def bs_power(d):
  return 0 - 10*2.9*np.log10(d)

power = [bs_power(x) for x in distances]

BERsdistance = []
SNRsdistance = []

Eb = 1
No = 1000
SNR = Eb/No

for i in power:
  # generate channel h(t) samples
  r, T = rayleighfade(10000, 20)
  theta = np.exp(-1j*np.random.uniform(0, 2*np.pi, 10000))
  h = r*theta
  n = np.random.normal(0, 1, 10000)*SNR*abs(i)
  SNRsdistance.append(SNR*abs(i))

  # recover signal
  cnoise = n/h
  eq_bits = bitsencoded + cnoise[:5000]

  # decode recieved signal
  eq_decoded = []

  for i in eq_bits:

    real = np.real(i)
    imag = np.imag(i)

    eq_decoded.append(ispositive(real*-1))
    eq_decoded.append(ispositive(imag))

  eq_decoded = np.array(eq_decoded)

  err = 0
  for i in range(len(bits)):

    if bits[i] != eq_decoded[i]:
      err += 1

  BER = err/10000
  BERsdistance.append(BER)

movepattern = np.append(np.negative(distances[:30]), distances[30:60])

plt.figure(figsize=(25, 10))

plt.xlabel('Distance')
plt.ylabel('BER')

plt.grid(True)
plt.title('Bit Error Rate vs. Mobile User Distance')
plt.plot(movepattern, BERsdistance)

plt.figure(figsize=(25, 10))

plt.xlabel('Distance')
plt.ylabel('SNR (dB)')

plt.grid(True)
plt.title('SNR vs. Mobile User Distance')
plt.plot(movepattern, -10*np.log10(SNRsdistance))