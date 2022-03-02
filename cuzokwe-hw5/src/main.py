# -*- coding: utf-8 -*-
"""ECET 512 HW 5

# Simulated Rayleigh Fading Signal
"""

import numpy as np
import matplotlib.pyplot as plt

# simulation parameters
N = 8192
fm20 = 20
fm200 = 200
fm = fm20

df = (2*fm)/(N-1)
T = 1/df
print(T)
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

#tX = np.insert(X, 4096, 0)
#tY = np.insert(Y, 4096, 0)

tX = np.abs(np.fft.ifft(X))**2
tY = np.abs(np.fft.ifft(Y))**2


#z=tX+1j*tY
#r=np.abs(z)

r = np.sqrt(tX+tY)
#print(len(test))

# Generating complex envelope
#z=tX+1j*tY
#env=np.abs(z)

# Plotting the envelope in the time domain 

plt.figure(figsize=(25, 10))

t=np.arange(0, T, T/N)
plt.plot(t, 10*np.log10((r/np.max(r))),'b')
plt.xlim(0, 1)
plt.ylim(-30, 0)

plt.xlabel('Time (s)')
plt.ylabel('Gain(dB)')

plt.grid(True)
plt.title('Normalized Gain in a Rayleigh Fading Channel fm = 200Hz')

plt.show()
plt.savefig("fading20fm.png")

r.shape

"""# Plotting Power Recieved from Basestations w/ Shadowing and Fading

### Parameters, Functions and Modules
"""

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
    
def drawCluster(center, N, radius):
  centers = []
  centers.append(center.copy())
  y = (radius * np.sqrt(3))/2
  x = radius
  tempcenter = center.copy()

  if N == 3:
    tempcenter[0] += 1.5*x
    tempcenter[1] -= y
    centers.append(tempcenter)

    tempcenter = center.copy()
    tempcenter[1] -= 2*y
    centers.append(tempcenter)

  if N == 4:
    tempcenter[0] += 1.5*x
    tempcenter[1] -= y
    centers.append(tempcenter)

    tempcenter = center.copy()
    tempcenter[1] -= 2*y
    centers.append(tempcenter)

    tempcenter = center.copy()
    tempcenter[0] += 1.5*x
    tempcenter[1] -= 3*y
    centers.append(tempcenter)

  if N == 7:
    xs = [0, 1.5*x, 1.5*x, 0, -1.5*x, -1.5*x,  0]
    cx = [x + center[0] for x in xs]

    ys = [0, -y, y, 2*y, y, -y, -2*y]
    cy = [y + center[1] for y in ys]

    centers = np.column_stack((cx, cy))

  return centers

def channelCenters(center, i, j, radius):
    yr = ((radius * np.sqrt(3))/2)*2
    pts = [1, 3, 5, 7, 9, 11]
    dpts = [3, 5, 7, 9, 11, 13]

    theta = list(map(hexangles,pts))
    dtheta = list(map(hexangles, dpts))

    xs = center[0]+(yr*np.cos(theta)*i)+(yr*np.cos(dtheta)*j)
    xs = np.insert(xs, 0, center[0])

    ys = center[1]+(yr*np.sin(theta)*i)+(yr*np.sin(dtheta)*j)
    ys = np.insert(ys, 0, center[0])

    k = np.column_stack((xs, ys))    
    return k

def findServingCell(mobileLocation, cellCenters):
  distances = [np.sqrt((mobileLocation[0]-cell[0])**2 + (mobileLocation[1]-cell[1])**2)  for cell in cellCenters]
  return distances.index(min(distances)), min(distances)

def dis(x1, y1, x2, y2):
  return np.sqrt((x1-x2)**2 + (y1-y2)**2)

"""## Animating a Mobile Handoff"""

# Set up plot
fig = plt.figure() #figsize=(20, 20))
ax = plt.gca()
ax.set_title('Animation of Inter-Cell Handoff')
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_aspect(1)
#plt.xlim(-300,300)
#plt.ylim(-300,300)

# Trajectory of Mobile Station
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
numFrames = 90
ims=[]
mobilePosX = np.linspace(0,1732.05080757,numFrames)
mobilePosY = np.linspace(0,1000,numFrames)


zippedcells = []
targetcells = []
k = drawCluster(center, N, radius)
for i in range(len(k)):
  drawCell(k[i],radius,labels[i]+subscripts[0])
  cell = [k[i], labels[i]+subscripts[0], 1/frequencies[i]]
  zippedcells.append(cell)

zippedcells = np.array(zippedcells, dtype=object)
#print(zippedcells[0:,0][0])

Adistances = []
Cdistances = []

for frames in range(numFrames):
   
   # Calculate distance from A1 
   distance = dis(mobilePosX[frames],mobilePosY[frames], zippedcells[0:,0][0][0], zippedcells[0:,0][0][1])
   Adistances.append(distance)

   distance = dis(mobilePosX[frames],mobilePosY[frames], zippedcells[0:,0][2][0], zippedcells[0:,0][2][1])
   Cdistances.append(distance)
   #targetcells.append(zippedcells[idx])
   # Draw a line connecting the center (basestation) of the serving cell 
   # and the mobile user
   #im, = plt.plot([0,mobilePosX[frames]],[0,mobilePosY[frames]], marker = 'x', color = 'red', animated=True)

   im = plt.plot( [zippedcells[0:,0][0][0],mobilePosX[frames]], [zippedcells[0:,0][0][1],mobilePosY[frames]], marker = 'x', color = 'red', animated=True)
   im2 = plt.plot( [zippedcells[0:,0][2][0],mobilePosX[frames]], [zippedcells[0:,0][2][1],mobilePosY[frames]], marker = 'x', color = 'blue', animated=True)
   # Draw the mobile user at the appropriate location
   #im2, = plt.plot(mobilePosX[frames],mobilePosY[frames],'r+', animated=True)
    
   ims.append(im+im2)

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

rc('animation', html='jshtml')
ani
ani.save('BSPower.gif', writer='pillow')
#plt.show()
#plt.clf()

"""### Plotting Power Recieved from Both Basestations"""

def bs_power(d):
  return 0 - 10*2.9*np.log10(d)

def bs_power_shadow(d):
  k = np.random.normal(0, 4, 1)
  return 0 - 10*2.9*np.log10(d) + k

Apower = list(map(bs_power, Adistances))
Apowershadow = list(map(bs_power_shadow, Adistances))
Cpower = list(map(bs_power, Cdistances))
Cpowershadow = list(map(bs_power_shadow, Cdistances))
fig = plt.figure(figsize=(20, 10))

plt.plot(Apower, color = 'red', label='A1 Power')
plt.plot(Apowershadow, marker = 'x', linestyle = ':', color = 'red', label='A1 Power + Shadow')
plt.plot(Cpower, color = 'blue', label='C1 Power')
plt.plot(Cpowershadow, marker = 'x', linestyle = ':', color = 'blue', label='C1 Power + Shadow')

a = np.array(Apower)

# plot handoff power and min power
plt.plot(np.full(shape=len(Apower), fill_value=Apower[np.argmax(a<-88)-4], dtype=np.int), linestyle = ':', color = 'black', label = 'Power at Handoff (Power Axis)')
plt.plot(np.full(shape=len(Apower), fill_value=-88, dtype=np.int), linestyle = '--', color = 'black', label = 'Minimum Usable Signal Level (Power Axis)')

# plot handoff interval time
plt.axvline(np.argmax(a<-88)-5, linestyle = ':', color = 'black', label='Handoff Initiated (Time Axis)')
plt.axvline(np.argmax(a<-88), linestyle = '--', color = 'black', label='Handoff Complete (Time Axis)')

plt.title('Power Recieved from Both Basestations in Inter-Cell Handoff')
plt.xlabel('Time (Seconds)')
plt.ylabel('Power (dBm)')

plt.ylim(-108,-40)

plt.legend(bbox_to_anchor=(1,1))
plt.show()

"""#### Without Shadowing and Fading Effects"""

fig = plt.figure(figsize=(25, 10))

plt.title('Power Recieved from Both Basestations in Inter-Cell Handoff (Shadowing and Fading Neglected)')
plt.xlabel('Time (Seconds)')
plt.ylabel('Power (dBm)')

plt.plot(Apower, color = 'red', label='A1 Power')
plt.plot(Cpower, color = 'blue', label='C1 Power')

plt.ylim(-110,-40)
plt.legend(bbox_to_anchor=(1,1))

"""#### Without Shadowing"""

fig = plt.figure(figsize=(25, 10))

plt.title('Power Recieved from Both Basestations in Inter-Cell Handoff (With Shadowing)')
plt.xlabel('Time (Seconds)')
plt.ylabel('Power (dBm)')

plt.plot(Apower, color = 'red', linestyle = ':', label='A1 Power')
plt.plot(Apowershadow, marker = 'x', color = 'red', label='A1 Power + Shadow')
plt.plot(Cpower, color = 'blue', linestyle = ':', label='C1 Power')
plt.plot(Cpowershadow, marker = 'x', color = 'blue', label='C1 Power + Shadow')

plt.ylim(-110,-40)
plt.legend(bbox_to_anchor=(1,1))

"""#### With Fading"""

Apowerfade = []
j = 0

for i, power in enumerate(Apower):
  Apowerfade.append(r[j]+power)

  while t[j] < i:

    #Apowerfade.append(r[j]+power)
    j = j+1

Cpowerfade = []
j = 0

for i, power in enumerate(Cpower):
  Cpowerfade.append(r[j]+power)

  while t[j] < i:
    #Cpowerfade.append(r[j]+power)
    j = j+1

print(len(Apowerfade))

fig = plt.figure(figsize=(25, 10))

plt.title('Power Recieved from Both Basestations in Inter-Cell Handoff (With Fading)')
plt.xlabel('Time (Seconds)')
plt.ylabel('Power (dBm)')

plt.plot(Apower, color = 'red', linestyle = ':', label='A1 Power')
plt.plot(Apowerfade, marker = 'x', color = 'red', label='A1 Power + Fading')
plt.plot(Cpower, color = 'blue', linestyle = ':', label='C1 Power')
plt.plot(Cpowerfade, marker = 'x', color = 'blue', label='C1 Power + Fading')
#plt.xlim(0, 200)
plt.ylim(-110,-40)
plt.legend(bbox_to_anchor=(1,1))

"""#### With Fade+Shadow"""

Apowerfade_shadow = []
j = 0

for i, power in enumerate(Apowershadow):
  Apowerfade_shadow.append(r[j]+power)

  while t[j] < i:

    #Apowerfade.append(r[j]+power)
    j = j+1

Cpowerfade_shadow = []
j = 0

for i, power in enumerate(Cpowershadow):
  Cpowerfade_shadow.append(r[j]+power)

  while t[j] < i:
    #Cpowerfade.append(r[j]+power)
    j = j+1

fig = plt.figure(figsize=(25, 10))

plt.title('Power Recieved from Both Basestations in Inter-Cell Handoff (With Shadowing and Fading)')
plt.xlabel('Time (Seconds)')
plt.ylabel('Power (dBm)')

plt.plot(Apower, color = 'red', linestyle = ':', label='A1 Power')
plt.plot(Apowerfade_shadow, marker = 'x', color = 'red', label='A1 Power + Shadow + Fading')
plt.plot(Cpower, color = 'blue', linestyle = ':', label='C1 Power')
plt.plot(Cpowerfade_shadow, marker = 'x', color = 'blue', label='C1 Power + Shadow + Fading')
#plt.xlim(0, 200)
plt.ylim(-110,-40)
plt.legend(bbox_to_anchor=(1,1))