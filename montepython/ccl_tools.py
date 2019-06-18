import re, os
import random
import numpy as np
import pymaster as nmt
import pyccl as ccl


default_pars = {
    'h'               :  0.67,
    'Omega_c'         :  0.27,
    'Omega_b'         :  0.045,
    # 'ln10_A_s'        :  3.044522,
    'sigma_8'          :  0.840421163375,
    'n_s'             :  0.96,
    'w_0'             : -1.0,
    'w_a'             :  0.0
}


def nofz(z,z0,sz,ndens):
    return np.exp(-0.5*((z-z0)/sz)**2)*ndens/np.sqrt(2*np.pi*sz**2)


def flatten_cls(cls, n_bte, n_ells):
    flat_cls = np.moveaxis(cls,[-3,-2,-1],[0,1,2])
    flat_cls = flat_cls[np.triu_indices(n_bte)]
    flat_cls = flat_cls.reshape(((n_bte+1)*n_bte*n_ells/2,)+cls.shape[:-3])
    return flat_cls


def get_cosmo_ccl(pars):
    cosmo = ccl.Cosmology(
        h        = pars['h'],
        Omega_c  = pars['Omega_c'],
        Omega_b  = pars['Omega_b'],
        sigma8   = pars['sigma_8'],
        # A_s      = (10.**(-10.))*np.exp(pars['ln10_A_s']),
        n_s      = pars['n_s'],
        w0       = pars['w_0'],
        wa       = pars['w_a']
        )
    ccl.linear_matter_power(cosmo,0.1,0.5)
    return cosmo


def get_tracers_ccl(cosmo, z, pz, bz):
    n_bins = pz.shape[0]
    # Tracers
    tracers = []
    for i in range(n_bins):
        tracers.append(
            ccl.NumberCountsTracer(cosmo,has_rsd=False,dndz=(z[i],pz[i]),bias=(z[i],bz[i]))
            )
        tracers.append(
            ccl.WeakLensingTracer(cosmo,dndz=(z[i],pz[i]))
            )
    return np.array(tracers)


def get_cls_ccl(cosmo, tracers, ell_bp):
    n_bte = tracers.shape[0]
    n_ells = len(ell_bp)
    cls = np.zeros([n_bte, n_bte, n_ells])

    for c1 in range(n_bte): # c1=te1+b1*n_te
        for c2 in range(c1, n_bte):
            cls[c1,c2,:] = ccl.angular_cl(cosmo,tracers[c1],tracers[c2],ell_bp)
            cls[c2,c1,:] = cls[c1,c2,:]
    cls_flat = flatten_cls(cls, n_bte, n_ells)
    return cls_flat


# Get data
dir = os.path.abspath('.')+'/data/covfefe/'

# Ells
ell_bp = np.load(os.path.join(dir, 'ell_bp.npz'))['lsims']

for n_bins in range(1,3):
    # Build photo_z
    z    = np.tile(np.linspace(0,3,512),[n_bins,1])
    cosmo = get_cosmo_ccl(default_pars)
    bz_ref=0.95*ccl.growth_factor(cosmo,1.)/ccl.growth_factor(cosmo,1./(1+z[0]))
    if n_bins==1:
        pz = np.array([
            nofz(z[0],0.955,0.13,7.55)
        ])
        bz = np.tile(0.65*bz_ref,[1,1])
    elif n_bins==2:
        pz = np.array([
            nofz(z[0],0.955,0.13,7.55),
            nofz(z[1],0.755,0.13,7.55)
        ])
        bz = np.tile(bz_ref,[2,1])
    np.savez_compressed(os.path.join(dir, 'z_{}'.format(n_bins)), z)
    np.savez_compressed(dir+'pz_{}'.format(n_bins), pz)
    np.savez_compressed(dir+'bz_{}'.format(n_bins), bz)

    # Build data
    tracers = get_tracers_ccl(cosmo, z, pz, bz)
    data = get_cls_ccl(cosmo, tracers, ell_bp)
    cov = np.load(os.path.join(dir, 'cov_sim_{}.npz'.format(n_bins)))['arr_0']
    L = np.linalg.cholesky(cov)
    u = np.random.randn(2*n_bins*(2*n_bins+1)/2*len(ell_bp))
    data = data# + L.dot(u)
    np.savez_compressed(dir+'cls_{}'.format(n_bins), data)
