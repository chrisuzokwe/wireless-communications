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


# function to determine cell sector label
def sector(user, cell):
  dy = user[1] - cell[1]
  dx = user[0] - cell[0]

  angle = np.arctan2(dy,dx) * 180 / np.pi

  if (angle <= 120) & (angle > 0):
    return '120' + u"\N{DEGREE SIGN}"
  elif (angle > 120) & (angle  <= 180):
    return '240' + u"\N{DEGREE SIGN}"
  elif (angle >= -180) & ( angle < -120):
    return '240' + u"\N{DEGREE SIGN}"
  elif (angle < 0)  & (angle >= -120):
    return '360' + u"\N{DEGREE SIGN}"


fig = plt.figure(figsize=(30, 15))

for i in range(len(zippedcells)):

  frameTotal120 = []
  frameTotal240 = []
  frameTotal360 = []

  for j in range(numFrames):
    fc120 = 0
    fc240 = 0
    fc360 = 0

    for user, k in enumerate(indexs[j]): 
      if k == i:
        
        label = sector([usersX[0:, user][j], usersY[0:, user][j]], zippedcells[i][0])

        if label == '120' + u"\N{DEGREE SIGN}":
          fc120 = fc120 + 1

        if label == '240' + u"\N{DEGREE SIGN}":
          fc240 = fc240 + 1

        if label == '360' + u"\N{DEGREE SIGN}":
          fc360 = fc360 + 1

    frameTotal120.append(fc120)
    frameTotal240.append(fc240)
    frameTotal360.append(fc360)

  if sum(frameTotal120) == 0:
    pass
  else:
    plt.plot(frameTotal120, label=zippedcells[i][1] + '120' + u"\N{DEGREE SIGN}")

  if sum(frameTotal240) == 0:
    pass
  else:
    plt.plot(frameTotal240, label=zippedcells[i][1] + '240' + u"\N{DEGREE SIGN}")

  if sum(frameTotal360) == 0:
    pass
  else:
    plt.plot(frameTotal360, label=zippedcells[i][1] + '360' + u"\N{DEGREE SIGN}")


plt.title('Count of Users per Cell vs Frame Data in 120 Degree Sectored Cell')
plt.xlabel('Frame Number (30 FPS)')
plt.ylabel('Total Users in Cell Sector')

plt.legend(bbox_to_anchor=(1,1))
plt.show()