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

ni = 1
nj = 2
N = ni**2 + ni*nj + nj**2
center = [0,0]
radius = 100
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
subscripts = ['\u2081', '\u2082', '\u2083', '\u2084', '\u2085', '\u2086', '\u2087']
frequencies = [1.8E9, 1.82E9, 1.84E9, 1.86E9, 1.88E9, 868E6, 882E6]

fig = plt.figure()#figsize=(20, 20))
ax = plt.gca()
ax.set_title('Multi-User Trunking Illustration')
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_aspect(1)
#plt.xlim(-300,300)
#plt.ylim(-300,300)


channels = channelCenters(center, ni, nj, radius)
zippedcells = []

# Draw 7 Tier Cell
for j in range(len(channels)):
  k = drawCluster(channels[j], N, radius)
  for i in range(len(k)):
    drawCell(k[i],radius,labels[i]+subscripts[j])
    cell = [k[i], labels[i]+subscripts[j], 1/frequencies[i]]
    zippedcells.append(cell)

plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
numFrames = 100
ims=[]
usersX = np.array([], dtype=np.int64).reshape(numFrames,0)
usersY = np.array([], dtype=np.int64).reshape(numFrames,0)

for i in range(20):
  mobilePosX = np.linspace(np.random.uniform(-1, 1)*400,np.random.uniform(-1, 1)*400,numFrames).reshape(numFrames, 1)
  mobilePosY = np.linspace(np.random.uniform(-1, 1)*400,np.random.uniform(-1, 1)*400,numFrames).reshape(numFrames, 1)

  usersX = np.hstack((usersX,mobilePosX))
  usersY = np.hstack((usersY, mobilePosY))

#print(usersX[0:,0])
zippedcells = np.array(zippedcells, dtype=object)
indexs = []

for frames in range(numFrames):

   im = []  
   idxs = []
   for i in range(20):
     
     idx, distance = findServingCell([usersX[0:, i][frames],usersY[0:, i][frames]], zippedcells[0:,0])
     idxs.append(idx)

     im = im + plt.plot( [zippedcells[idx,0][0],usersX[0:, i][frames]], [zippedcells[idx,0][1],usersY[0:, i][frames]], marker = 'x', color = 'blue', animated=True)
   
   indexs.append(idxs)
   ims.append(im)


#print(indexs)
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

rc('animation', html='jshtml')
ani
ani.save('RandomUserCellMvmt.gif', writer='pillow')
#plt.show()

#plt.show()
#plt.clf()

fig = plt.figure(figsize=(20, 10))

for i in range(len(zippedcells)):
  frameTotal = []
  for j in range(numFrames):
    frameTotal.append(indexs[j].count(i))

  if sum(frameTotal) == 0: continue
  plt.plot(frameTotal, label=zippedcells[i][1])

plt.title('Count of Users per Cell vs Frame Data')
plt.xlabel('Frame Number (30 FPS)')
plt.ylabel('Total Users in Cell')

plt.legend(bbox_to_anchor=(1,1))
plt.show()