# Chris Uzokwe - ECET 512 - 1/20/2021


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, rc
from IPython.display import HTML
import cmath

ni = 1
nj = 2
N = ni**2 + ni*nj + nj**2
center = [0,0]
radius = 100
labels = ['A', 'B', 'C', 'D', 'E,', 'F', 'G']
subscripts = ['\u2081', '\u2082', '\u2083', '\u2084', '\u2085', '\u2086', '\u2087']

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
  return distances.index(min(distances))

channels = channelCenters(center, ni, nj, radius)

findServingCell([400,400], channels)

fig = plt.figure()#figsize=(20, 20))
ax = plt.gca()
ax.set_title('Cell Illustration (N=7)')
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_aspect(1)
#plt.xlim(-300,300)
#plt.ylim(-300,300)

for j in range(len(channels)):
  k = drawCluster(channels[j], N, radius)
  for i in range(len(k)):
    drawCell(k[i],radius,labels[i]+subscripts[j])

plt.show()
plt.clf()




## Animate User

fig = plt.figure()
ax = plt.gca()
ax.set_aspect(1)
#plt.xlim(-300,300)
#plt.ylim(-300,300)
ax.set_title('Cell Mobile User Animation')
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')

plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
numFrames = 60
ims=[]
mobilePosX = np.linspace(-500,500,numFrames)
mobilePosY = np.linspace(-500,500,numFrames)

for frames in range(numFrames):
   channels = channelCenters(center, ni, nj, radius)
   all = channels.copy()
   # Draw the serving cells and label them
   for j in range(len(channels)):
     k = drawCluster(channels[j], N, radius)
     all = np.concatenate((all, k))
     #print(j)
     for i in range(len(k)):
       drawCell(k[i],radius,labels[i]+subscripts[j])
 
   # Find the corresponding serving cell
   idx = findServingCell([mobilePosX[frames],mobilePosY[frames]], all)
   #print(channels[idx][0])

   # Draw a line connecting the center (basestation) of the serving cell 
   # and the mobile user
   #im, = plt.plot([0,mobilePosX[frames]],[0,mobilePosY[frames]], marker = 'x', color = 'red', animated=True)

   im, = plt.plot( [all[idx][0],mobilePosX[frames]], [all[idx][1],mobilePosY[frames]], marker = 'x', color = 'red', animated=True)

   # Draw the mobile user at the appropriate location
   #im2, = plt.plot(mobilePosX[frames],mobilePosY[frames],'r+', animated=True)
    
   ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

rc('animation', html='jshtml')
ani
#ani.save('drawCell.gif', writer='pillow')
#plt.show()