import os
import numpy as np
from montepython.likelihood_class import Likelihood
import montepython.io_mp as io_mp
import warnings
from astropy.io import fits
import montepython.ccl_tools as tools


class CFHTLens_KL_badfit(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # Read photo_z data
        with fits.open(os.path.join(self.data_directory, self.file)) as fn:
            self.photo_z = fn['PHOTO_Z'].data

        self.is_kl = False
        # Read method
        self.method = data.cosmo_arguments['method']
        test = self.method in ['full', 'kl_diag', 'kl_off_diag']
        assert test, 'Method not recognized! Options are: full, kl_off_diag or kl_diag'

        # Read kl_arguments
        if self.method in ['kl_diag', 'kl_off_diag']:

            self.is_kl = True

            self.n_kl = data.cosmo_arguments['n_kl']
            test = self.n_kl>0 and self.n_kl<self.photo_z.shape[0]
            assert test, 'n_kl should be greater than 0 and smaller than {}'.format(self.photo_z.shape[0])

            self.kl_scale_dep = tools.read_bool(data.cosmo_arguments['kl_scale_dep'])


        if self.method == 'kl_diag':
            self.is_diag = True
        else:
            self.is_diag = False


        # Read data
        self.mask_ell = np.array(self.mask_ell)
        self.bandpowers = np.array(self.bandpowers)
        with fits.open(os.path.join(self.data_directory, self.file)) as fn:
            self.ell = fn['ELL'].data
            self.cls = fn['CL_EE'].data
            self.noise = fn['CL_EE_NOISE'].data
            self.sims = fn['CL_SIM_EE'].data
            if self.method in ['kl_diag', 'kl_off_diag']:
                if self.kl_scale_dep:
                    self.kl_t = fn['KL_T_ELL'].data
                else:
                    self.kl_t = fn['KL_T'].data

        # How many sims
        self.n_data = len(self.mask_ell[self.mask_ell])
        self.n_bins = self.photo_z.shape[0]-1
        self.n_data_tot = self.n_data*self.n_bins*(self.n_bins+1)/2
        if self.method == 'kl_diag':
            self.n_data = self.n_data*self.n_kl
        elif self.method == 'kl_off_diag':
            self.n_data = self.n_data*self.n_kl*(self.n_kl+1)/2
        else:
            self.n_data = self.n_data*self.n_bins*(self.n_bins+1)/2
        self.n_sims = tools.how_many_sims(data.cosmo_arguments['n_sims'], self.sims.shape[1], self.n_data, self.n_data_tot)


        # Clean Cl's
        self.cls = tools.clean_cl(self.cls, self.noise)
        self.sims = tools.clean_cl(self.sims, self.noise)


        # Apply KL
        if self.method in ['kl_off_diag', 'kl_diag']:
            self.cls = tools.apply_kl(self.kl_t, self.cls, self.method, self.kl_scale_dep, self.n_kl)
            self.sims = tools.apply_kl(self.kl_t, self.sims, self.method, self.kl_scale_dep, self.n_kl)


        # Select sims
        self.sims = tools.select_sims(self.sims, self.n_sims)

        # Unify fields
        self.cov_pf = tools.get_covmat(self.sims, self.is_diag)
        self.cls = tools.unify_fields_cl(self.cls, self.cov_pf, self.is_diag)
        self.sims = tools.unify_fields_cl(self.sims, self.cov_pf, self.is_diag)


        # Mask Cl's
        self.cls = tools.mask_cl(self.cls, self.mask_ell, self.is_diag)
        self.sims = tools.mask_cl(self.sims, self.mask_ell, self.is_diag)


        # Calculate covmat Cl's
        cov = tools.get_covmat(self.sims, self.is_diag)


        # Flatten Cl's and covmat
        self.cls = tools.flatten_cl(self.cls, self.is_diag)
        cov = tools.flatten_covmat(cov, self.is_diag)


        # Calculate inverse of covmat
        factor = (self.n_sims-self.n_data-2.)/(self.n_sims-1.)
        self.inv_cov_mat = factor*np.linalg.inv(cov)


        self.mcm_path = os.path.join(self.data_directory, self.mcm)

        # end of initialization



    # compute likelihood

    def loglkl(self, cosmo, data):

        if self.method in ['kl_off_diag', 'kl_diag']:
            theory_cls = cosmo.get_theory_cl(self.photo_z,
                self.bandpowers,
                self.mcm_path,
                self.method,
                self.is_diag,
                self.cov_pf,
                self.mask_ell,
                kl_t=self.kl_t,
                kl_scale_dep=self.kl_scale_dep,
                n_kl=self.n_kl)
        else:
            theory_cls = cosmo.get_theory_cl(self.photo_z,
                self.bandpowers,
                self.mcm_path,
                self.method,
                self.is_diag,
                self.cov_pf,
                self.mask_ell)

        # Calculate Gaussian prior for Intrinsic Alignement
        if cosmo.pars['A_IA'] !=0:
            lp = -(cosmo.pars['beta_IA']-cosmo.pars['beta_IA_prior'][0])**2./2./cosmo.pars['beta_IA_prior'][1]**2.
        else:
            lp = 0.

        #Get chi2
        chi2 = (self.cls-theory_cls).dot(self.inv_cov_mat).dot(self.cls-theory_cls)


        lkl = lp - 0.5 * chi2

        return lkl
