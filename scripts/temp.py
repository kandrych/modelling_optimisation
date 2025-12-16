import numpy as np
import pymcfost
import matplotlib.pyplot as plt

# Load one model directory (thermal run)
model = pymcfost.Grid("/fred/oz061/kandrych/Aksita/AR_Pup/original/data_th")   # or whatever your folder is called

# Grid coordinates
r = model.grid.r        # [nr]
z = model.grid.z        # [nz]
# dust_temperature has shape [ndust, nr, nz, nphi] in many versions
T_dust = model.dust_temperature[0]   # first dust population

# Find midplane index (z ~ 0)
mid_idx = np.argmin(np.abs(z))

# Average over azimuth φ at midplane
T_mid = T_dust[:, mid_idx, :].mean(axis=-1)   # [nr]

# Plot T(r)
plt.plot(r, T_mid)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("r [AU]")
plt.ylabel("T_midplane [K]")
plt.show()
plt.savefig("midplane_temperature.png")
