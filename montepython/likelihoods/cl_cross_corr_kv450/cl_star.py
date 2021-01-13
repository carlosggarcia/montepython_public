#!/usr/bin/python
from scipy.special import jv
import numpy as np
import scipy.integrate as integrate

def get_Cl_star(ells):
    fname = '/mnt/extraspace/damonge/S8z_data/KiDS_data/KV450_COSMIC_SHEAR_DATA_RELEASE/SUPPLEMENTARY_FILES/KV450_xi_pm_c_term.dat'
    theta, xip_c_per_zbin, xim_c_per_zbin = np.loadtxt(fname, usecols=(1, 3, 4), unpack=True)
    integrand = integrate.simps(theta * xip_c_per_zbin * jv(0, ells[:, None] * theta), theta, axis=1)
    Cl_star = 2 * np.pi * integrand
    return Cl_star

ells = np.array([20, 40, 60, 80])
cl_star = get_Cl_star(ells)
print(cl_star.shape)
np.savetxt('cl_star.txt', np.array([ells, cl_star]).T)
