def friispathloss(wavelength, d):
  return (wavelength/(4*np.pi*d))**2

def exppathloss(d, B):
  return 1E-3 - 10*B*np.log10(d)

e = np.random.normal(0, 8, 1)

def expshadowpathloss(d, B):
  return 1E-3 - 10*B*np.log10(d) + e

targetcells = np.array(targetcells)
powers = np.array(list(map(friispathloss, targetcells[0:, 2], distances)))

powers.resize((60,1))
targetcells = np.concatenate((targetcells, powers), axis=1)

cellnum = targetcells[0][1]
splits = []
div = 0

for i in range(len(targetcells)):
  if targetcells[i][1] == cellnum:
    div += 1
  else: 
    splits.append(div)
    div = 1
    cellnum = targetcells[i][1]

idx = 0
for i in splits:
  plt.plot(range(idx,idx+i), targetcells[idx:idx+i,3:4], 'x', linestyle='-' ,label=targetcells[idx][1])
  idx = idx+i

plt.title('Recieved Power vs. Time')
plt.xlabel('Frame Number (30 FPS)')
plt.ylabel('Recieved Power [W]')

plt.legend()
plt.show()

p3 = np.array(list(map(exppathloss, distances, np.full(
    shape=len(distances),
    fill_value=3,
    dtype=np.int))))

p3s = np.array(list(map(expshadowpathloss, distances, np.full(
    shape=len(distances),
    fill_value=3,
    dtype=np.int))))

p4 = np.array(list(map(exppathloss, distances, np.full(
    shape=len(distances),
    fill_value=4,
    dtype=np.int))))

p4s = np.array(list(map(expshadowpathloss, distances, np.full(
    shape=len(distances),
    fill_value=4,
    dtype=np.int))))

cellnum = targetcells[0][1]
splits = []
div = 0

for i in range(len(targetcells)):
  if targetcells[i][1] == cellnum:
    div += 1
  else: 
    splits.append(div)
    div = 1
    cellnum = targetcells[i][1]

splits.append(div)

idx = 0
for i in splits:
  plt.plot(range(idx,idx+i), np.zeros(idx+i-idx)-100, 'x', label=targetcells[idx][1])
  idx = idx+i

plt.plot(p3, linestyle = '--', label = 'Exp 3')
plt.plot(p3s, linestyle=':', label = 'Exp 3 + Shadow')
plt.plot(p4, linestyle = '-.', label = 'Exp 4')
plt.plot(p4s, label = 'Exp 4 + Shadow')
plt.title('Downlink Signal Power vs. Time')
plt.xlabel('Frame Number (30 FPS)')
plt.ylabel('Downlink Signal Power [dBm]')

plt.legend(bbox_to_anchor=(1,1))
plt.show()

fig = plt.figure()
ax = plt.gca()
ax.set_aspect(1)
#plt.xlim(-300,300)
#plt.ylim(-300,300)
ax.set_title('Cell Mobile User Animation - First Tier Downlink Interference')
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')

plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
numFrames = 60
ims=[]
mobilePosX = np.linspace(-170,170,numFrames)
mobilePosY = np.linspace(-170,170,numFrames)

channels = channelCenters(center, ni, nj, radius)
all = channels.copy()

zippedcells = []
distances = []
targetcells = []

# Draw the serving cells and label them
for j in range(len(channels)):
  k = drawCluster(channels[j], N, radius)
  for i in range(len(k)):
    drawCell(k[i],radius,labels[i]+subscripts[j])
    cell = [k[i], labels[i]+subscripts[j], 1/frequencies[i]]
    zippedcells.append(cell)

zippedcells = np.array(zippedcells, dtype=object)

for frames in range(numFrames):
   
   # Find the corresponding serving cell
   idx, distance = findServingCell([mobilePosX[frames],mobilePosY[frames]], zippedcells[0:,0])
   #print(channels[idx][0])
   distances.append(distance)
   targetcells.append(zippedcells[idx])
   # Draw a line connecting the center (basestation) of the serving cell 
   # and the mobile user
   #im, = plt.plot([0,mobilePosX[frames]],[0,mobilePosY[frames]], marker = 'x', color = 'red', animated=True)

   im = plt.plot( [zippedcells[idx+7,0][0],mobilePosX[frames]], [zippedcells[idx+7,0][1],mobilePosY[frames]], marker = 'x', color = 'red', animated=True)
   im2 = plt.plot( [zippedcells[idx+14,0][0],mobilePosX[frames]], [zippedcells[idx+14,0][1],mobilePosY[frames]], marker = 'x', color = 'green', animated=True)
   im3 = plt.plot( [zippedcells[idx+21,0][0],mobilePosX[frames]], [zippedcells[idx+21,0][1],mobilePosY[frames]], marker = 'x', color = 'blue', animated=True)
   im4 = plt.plot( [zippedcells[idx+28,0][0],mobilePosX[frames]], [zippedcells[idx+28,0][1],mobilePosY[frames]], marker = 'x', color = 'red', animated=True)
   im5 = plt.plot( [zippedcells[idx+35,0][0],mobilePosX[frames]], [zippedcells[idx+35,0][1],mobilePosY[frames]], marker = 'x', color = 'green', animated=True)
   im6 = plt.plot( [zippedcells[idx+42,0][0],mobilePosX[frames]], [zippedcells[idx+42,0][1],mobilePosY[frames]], marker = 'x', color = 'blue', animated=True)

   # Draw the mobile user at the appropriate location
   #im2, = plt.plot(mobilePosX[frames],mobilePosY[frames],'r+', animated=True)
   ims.append(im+im2+im3+im4+im5+im6)

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

rc('animation', html='jshtml')
ani
ani.save('drawCellFTI.gif', writer='pillow')
#plt.show()

def dis(x1, y1, x2, y2):
  return np.sqrt((x1-x2)**2 + (y1-y2)**2)

idxs = []

for frames in range(numFrames):
   # Find the corresponding serving cell
   idx, distance = findServingCell([mobilePosX[frames],mobilePosY[frames]], zippedcells[0:,0])
   idxs.append(idx)
   
   
   #distances.append(distance)
   #targetcells.append(zippedcells[idx])

p3iarray = []
p3isarray = []
p4iarray = []
p4isarray = []


for i, num in enumerate(idxs):
  totalpower = 0
  p3i = 0
  p3is = 0
  p4i = 0
  p4is = 0
  print(zippedcells[num][1])
  for j in range(1, 7):
    d = dis(mobilePosX[i], mobilePosY[i], zippedcells[0:,0][num+j*7][0], zippedcells[0:,0][num+j*7][1])
    
    p3i = p3i + exppathloss(d, 3)
    print(d, ':', p3i)
    p3is = p3is + expshadowpathloss(d, 3)
    p4i = p4i + exppathloss(d, 4)
    p4is = p4is + expshadowpathloss(d, 4)

  print(p3i)
  p3iarray.append(p3i)
  p3isarray.append(p3is)
  p4iarray.append(p4i)
  p4isarray.append(p4is)

cellnum = targetcells[0][1]
splits = []
div = 0
print(len(targetcells))

for i in range(len(targetcells)):
  if targetcells[i][1] == cellnum:
    div += 1

  else: 
    splits.append(div)
    div = 1
    cellnum = targetcells[i][1]

splits.append(div)
idx = 0
print(splits)
for i in splits:
  plt.plot(range(idx,idx+i), np.zeros(idx+i-idx)-700, 'x', label=targetcells[idx][1])
  idx = idx+i

plt.plot(p3iarray, linestyle = '--', label = 'Exp 3')
plt.plot(p3isarray, linestyle=':', label = 'Exp 3 + Shadow')
plt.plot(p4iarray, linestyle = '-.', label = 'Exp 4')
plt.plot(p4isarray, label = 'Exp 4 + Shadow')
plt.title('Downlink Interference Power vs. Time')
plt.xlabel('Frame Number (30 FPS)')
plt.ylabel('Downlink Iterference Power [dBm]')

plt.legend(bbox_to_anchor=(1,1))
plt.show()

p3 = np.array(list(map(exppathloss, distances, np.full(
    shape=len(distances),
    fill_value=3,
    dtype=np.int))))

p3s = np.array(list(map(expshadowpathloss, distances, np.full(
    shape=len(distances),
    fill_value=3,
    dtype=np.int))))

p4 = np.array(list(map(exppathloss, distances, np.full(
    shape=len(distances),
    fill_value=4,
    dtype=np.int))))

p4s = np.array(list(map(expshadowpathloss, distances, np.full(
    shape=len(distances),
    fill_value=4,
    dtype=np.int))))

idx = 0
for i in splits:
  plt.plot(range(idx,idx+i), np.zeros(idx+i-idx), 'x', label=targetcells[idx][1])
  idx = idx+i

plt.plot(p3/p3iarray, linestyle = '--', label = 'Exp 3')
plt.plot(p3s/p3isarray, linestyle=':', label = 'Exp 3 + Shadow')
plt.plot(p4/p4iarray, linestyle = '-.', label = 'Exp 4')
plt.plot(p4s/p4isarray, label = 'Exp 4 + Shadow')
plt.title('Downlink SIR vs. Time')
plt.xlabel('Frame Number (30 FPS)')
plt.ylabel('Downlink Ratio')

plt.legend(bbox_to_anchor=(1,1))
plt.show()