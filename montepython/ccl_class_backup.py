import re
import random
import numpy as np
import io_mp
import pyccl as ccl
import ccl_tools as tools



class CCL():
    """
    General class for the CCL object.

    """

    def __init__(self):

        self.state = 1

        # Set default parameters as in Joudaki et al. (2017)
        self.pars = {
            'h'               :  0.61197750,
            'omega_c'         :  0.11651890,
            'omega_b'         :  0.03274485,
            'ln10_A_s'        :  2.47363700,
            'n_s'             :  1.25771300,
            'w_0'             : -1.00000000,
            'w_A'             :  0.00000000,
            'A_IA'            :  0.00000000,
            'beta_IA'         :  1.13000000,
            'rho_crit_IA'     :  2.77536627e11,
            'C_1_IA'          :  5.e-14,
            'L_I_over_L_0_IA' :  np.array([0.017, 0.069, 0.15, 0.22, 0.36, 0.49, 0.77]),
            'beta_IA_prior'   :  np.array([1.13, 0.25])
        }


    def get_cosmo_ccl(self):

        cosmo_ccl = ccl.Cosmology(
            h        = self.pars['h'],
            Omega_c  = self.pars['omega_c']/self.pars['h']**2.,
            Omega_b  = self.pars['omega_b']/self.pars['h']**2.,
            A_s      = (10.**(-10.))*np.exp(self.pars['ln10_A_s']),
            n_s      = self.pars['n_s'],
            w0       = self.pars['w_0'],
            wa       = self.pars['w_A']
            )
        ccl.linear_matter_power(cosmo_ccl,0.1,0.5)

        return cosmo_ccl


    def get_sigma8(self):

        cosmo_ccl = self.get_cosmo_ccl()

        return ccl.sigma8(cosmo_ccl)


    def get_Omegam(self):

        Omm = (self.pars['omega_c']+self.pars['omega_b'])/self.pars['h']**2.

        return Omm


    def get_S8(self):

        S8 = self.get_sigma8()*(self.get_Omegam()/0.3)**(0.5)

        return S8


    def get_cls_ccl(self, cosmo_ccl, photo_z, ell_max):

        # Local variables
        n_bins = photo_z.shape[0]-1

        # z and pz
        z = photo_z[0].astype(np.float64)
        pz = photo_z[1:].astype(np.float64)

        # If Intrinsic Alignement
        if self.pars['A_IA'] != 0.:
            f_z = np.ones(len(z))
            # Bias
            Omega_m = (self.pars['omega_c']+self.pars['omega_b'])/self.pars['h']**2.
            D_z = ccl.background.growth_factor(cosmo_ccl, 1./(1.+z))
            b_z = -self.pars['A_IA']*self.pars['C_1_IA']*self.pars['rho_crit_IA']*Omega_m/D_z
            b_z = np.outer(self.pars['L_I_over_L_0_IA']**self.pars['beta_IA'], b_z)
            # Tracers
            lens = np.array([
                ccl.WeakLensingTracer(
                    cosmo_ccl,
                    dndz=(z,pz[x]),
                    ia_bias=(z,b_z[x]),
                    red_frac=(z,f_z),
                ) for x in range(n_bins)])
        else:
            # Tracers
            lens = np.array([
                ccl.WeakLensingTracer(
                    cosmo_ccl,
                    dndz=(z,pz[x])
                ) for x in range(n_bins)])

        # Cl's
        ell = np.arange(ell_max+1)
        cls = np.zeros((n_bins, n_bins, ell_max+1))
        for count1 in range(n_bins):
            for count2 in range(count1,n_bins):
                cls[count1,count2] = ccl.angular_cl(cosmo_ccl, lens[count1], lens[count2], ell)
                cls[count2,count1] = cls[count1,count2]
        cls = np.transpose(cls,axes=[2,0,1])

        return cls


    def get_theory_cl(self, photo_z, bp, mcm, method, is_diag, cov_pf, mask_ell, kl_t=None,kl_scale_dep=None,n_kl=None):

        cosmo_ccl = self.get_cosmo_ccl()
        cls = self.get_cls_ccl(cosmo_ccl, photo_z, bp[-1,-1])


        # Local variables
        ell = np.arange(bp[-1,-1]+1)
        n_bins = photo_z.shape[0]-1

        # Couple decouple Cl's
        cls = tools.couple_decouple_cl(ell, cls, mcm, n_bins, len(bp))

        # Apply KL
        if method in ['kl_off_diag', 'kl_diag']:
            cls = tools.apply_kl(kl_t, cls, method, kl_scale_dep, n_kl)

        # Unify fields Cl's
        cls = tools.unify_fields_cl(cls, cov_pf, is_diag)
        # Mask Cl's
        cls = tools.mask_cl(cls, mask_ell, is_diag)
        # Flatten Cl's
        cls = tools.flatten_cl(cls, is_diag)

        return cls


    def struct_cleanup(self):
        return

    def empty(self):
        return

    # Set up the dictionary
    def set(self,*pars_in,**kars):
        if len(pars_in)==1:
            self.pars.update(dict(pars_in[0]))
        elif len(pars_in)!=0:
            raise RuntimeError("bad call")
        self.pars.update(kars)
        return True


    def compute(self, level=[]):
        return

    def get_current_derived_parameters(self, names):

        derived = {}
        for name in names:
            if name == 'sigma_8':
                value = self.get_sigma8()
            elif name == 'Omega_m':
                value = self.get_Omegam()
            elif name == 'S_8':
                value = self.get_S8()
            else:
                raise RuntimeError("%s was not recognized as a derived parameter" % name)
            derived[name] = value

        return derived
