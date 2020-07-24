import os
import numpy as np
from montepython.likelihood_class import Likelihood
import montepython.io_mp as io_mp
import warnings
import ccl_tools as tools
import pyccl as ccl



class covfefe(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        self.nb = data.cosmo_arguments['n_bins']
        self.cm = data.cosmo_arguments['cov']
        n_sims = 20000

        # Load Covariance matrix
        fn = 'cov_{}_{}.npz'.format(self.cm,self.nb)
        self.cov = np.load(os.path.join(self.data_directory, fn))['arr_0']
        if self.cm=='sim':
            factor = (n_sims-self.cov.shape[0]-2.)/(n_sims-1.)
        else:
            factor = 1.
        self.icov = factor*np.linalg.inv(self.cov)

        # Load ell bandpowers
        self.ell_bp = np.load(os.path.join(self.data_directory, 'ell_bp.npz'))['lsims'].astype(int)
        self.nl = len(self.ell_bp)

        # Load photo_z
        fn = 'z_{}.npz'.format(self.nb)
        self.z = np.load(os.path.join(self.data_directory, fn))['arr_0']
        fn = 'pz_{}.npz'.format(self.nb)
        self.pz = np.load(os.path.join(self.data_directory, fn))['arr_0']
        fn = 'bz_{}.npz'.format(self.nb)
        self.bz = np.load(os.path.join(self.data_directory, fn))['arr_0']

        # Load data
        fn = 'data_{}.npz'.format(self.nb)
        self.data = np.load(os.path.join(self.data_directory, fn))['arr_0']

        # end of initialization



    # compute likelihood

    def loglkl(self, cosmo, data):

        # Get theory Cls
        cosmo_ccl = tools.get_cosmo_ccl(cosmo.pars)
        tracers = tools.get_tracers_ccl(cosmo_ccl, self.z, self.pz, self.bz)
        theory = tools.get_cls_ccl(cosmo_ccl, tracers, self.ell_bp)

        # Get chi2
        chi2 = (self.data-theory).dot(self.icov).dot(self.data-theory)
        lkl = - 0.5 * chi2

        return lkl
