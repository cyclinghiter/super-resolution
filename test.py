import itertools
import sys
import xarray as xr

import numpy as np 
import matplotlib.pyplot as plt

import utils
import function
import obj
import system

xd = xr.load_dataset("total_200828_68pt_distance_10-450_smm_pico.nc", engine="h5netcdf")
# load used patterns in experiment
patterns = []
for i in range(68):
    pattern = np.asarray([int(obj) for obj in list(xd.pattern.data[i][2:])]).reshape(20,8).T
    patterns.append(pattern)

# wave parameters
GHz = 1e9
freq = 2.7 * GHz
lamb = 3e8 / freq
k0 = 2*np.pi / lamb
kx = 0

pattern = np.array([1,1,1,1, 0,0,0,0, 1,1,1,1, 0,0,0,0, 1,1,1,1])

pattern = pattern[np.newaxis,:] * np.ones((8,1))
# pattern = patterns[32]
mySystem = system.Container(k0)
myArray = obj.SMMarray(0,0,1, 0.02, 0.02, amp=0.7, k0=k0, pattern = pattern)
antenna = obj.Antenna(0,0,0,1)
PWG = obj.PlaneWaveGenerator(0,0,0, kx=kx, ky=0, kz=np.sqrt(k0**2-kx**2))

myArray.shape
x = np.linspace(-0.3,0.3,50)
z = np.linspace(0,1.01,100)
y = 0
FR = obj.FieldRecorder(x,y,z)


mySystem.append(antenna)
mySystem.append(myArray)
mySystem.append(FR)
mySystem.append(PWG)


mySystem.experiment(source=PWG, record=True)

utils.imshow(np.log(np.abs(np.real(FR.E[0,0]))), vmin=-4)

# plt.imshow(np.real(FR.E[,0]))