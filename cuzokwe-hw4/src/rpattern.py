def rpattern(R, theta, phi):
  Eo = 0.8
  l = (3E8)/(900E6)
  r = 2*l*R
  n = 377

  A = 0.5*l
  B = 0.25*l
  a = 0.25*l
  b = 0.25*l

  k = (2*np.pi)/l

  Vx = A/l * np.sin(theta) * np.cos(phi)
  Vy = B/l * np.sin(theta) * np.sin(phi)

  F1 = 4/np.pi * (np.cos(np.pi*Vx)/(1-4*Vx**2))
  F0 = 2/np.pi * (np.sin(np.pi*Vy)/Vy)

  Etheta = 1j * (np.exp(-1*k*r*1j))/(l*r) * Eo * (A*B/4) * ((1+np.cos(theta))/2) * np.sin(phi) * F1 * F0
  Ephi = 1j * (np.exp(-1*k*r*1j))/(l*r) * Eo * (A*B/4) * ((1+np.cos(theta))/2) * np.cos(phi) * F1 * F0

  U = r**2 * (abs(Etheta)**2 + abs(Ephi)**2) / (2*n)
  return U


theta = np.arange(0.0001, 2*np.pi, 0.063)
phi = np.arange(0.0001, 2*np.pi, 0.063)
r = 2*(3E8)/(900E6)

theta, phi = np.meshgrid(theta, phi)

X = r*np.sin(theta)*np.cos(phi)
Y = r*np.sin(theta)*np.sin(phi)
Z = r*np.cos(theta)

r_intensity = [10*np.log10(rpattern(10, theta, phi)) for theta, phi in zip(theta, phi)]

r_intensity = np.array(r_intensity)
r_intensity.shape

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# plot 3D

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')

plt.title('Radiation Pattern of Horn Antenna at Fixed Length')
plt.xlabel('X')
plt.ylabel('Y')
ax.set_zlabel('Z')


ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.grid(True)

img = ax.scatter(X, Y, r_intensity, c=r_intensity, cmap=plt.jet())
cb = fig.colorbar(img)
cb.ax.set_ylabel('dB')
plt.show()

#Azimuth
# plot 3D

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)

plt.title('Radiation Pattern of Horn Antenna at Fixed Length (Azumith Plane)')
plt.xlabel('X')
plt.ylabel('Y')

ax.set_xticklabels([])
ax.set_yticklabels([])

img = ax.scatter(X, Y, c=r_intensity, cmap=plt.jet())
cb = fig.colorbar(img)
cb.ax.set_ylabel('dB')
plt.show()

# Elevation
# plot 3D

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)

plt.title('Radiation Pattern of Horn Antenna at Fixed Length (Elevation Plane)')
plt.xlabel('Z')
plt.ylabel('Y')

ax.set_xticklabels([])
ax.set_yticklabels([])

img = ax.scatter(X, r_intensity, c=r_intensity, cmap=plt.jet())
cb = fig.colorbar(img)
cb.ax.set_ylabel('dB')
plt.show()