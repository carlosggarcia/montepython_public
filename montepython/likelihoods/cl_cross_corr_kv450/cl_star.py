#!/usr/bin/python
from scipy.special import jv
import numpy as np
import scipy.integrate as integrate

def get_Cl_star(ells):
    fname = '/mnt/extraspace/damonge/S8z_data/KiDS_data/KV450_COSMIC_SHEAR_DATA_RELEASE/SUPPLEMENTARY_FILES/KV450_xi_pm_c_term.dat'
    theta, xip_c_per_zbin, xim_c_per_zbin = np.loadtxt(fname, usecols=(1, 3, 4), unpack=True)
    theta = np.radians(theta / 60.)
    integrand = integrate.simps(theta * xip_c_per_zbin * jv(0, ells[:, None] * theta), theta, axis=1)
    Cl_star = 2 * np.pi * integrand
    return Cl_star

bpw_edges = [0, 30, 60, 90, 120, 150, 180, 210, 240, 272, 309, 351, 398, 452, 513, 582, 661, 750, 852, 967, 1098, 1247, 1416, 1608, 1826, 2073, 2354, 2673, 3035, 3446, 3914, 4444, 5047, 5731, 6508, 7390, 8392, 9529, 10821, 12288]
ells = bpw_edges[:-1] + np.diff(bpw_edges) / 2.
cl_star = get_Cl_star(ells)
print(cl_star.shape)
np.savetxt('cl_star.txt', np.array([ells, cl_star]).T)
