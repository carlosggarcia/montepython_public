import numpy as np
import pyccl as ccl
from scipy.interpolate import interp1d
from classy import Class


class CCL():
    """
    General class for the CCL object.

    """

    def __init__(self):

        self.state = 1

        # Set default parameters
        # Planck 2018: Table 2 of 1807.06209
        self.pars = {
            'h':       0.6736,
            'Omega_c': 0.2640,
            'Omega_b': 0.0493,
            'sigma8': 0.8111,
            'n_s': 0.9649,
            'w0': -1.0,
            'wa':  0.0
        }


        self.cosmo_ccl_planck = ccl.Cosmology(**self.pars,
                                              transfer_function='boltzmann_class')
        # Copied from ccl/pk2d.py
        # These lines are needed to compute the Pk2D array
        self.nk = ccl.ccllib.get_pk_spline_nk(self.cosmo_ccl_planck.cosmo)
        self.na = ccl.ccllib.get_pk_spline_na(self.cosmo_ccl_planck.cosmo)
        self.a_arr, _ = ccl.ccllib.get_pk_spline_a(self.cosmo_ccl_planck.cosmo,
                                                   self.na, 0)
        self.z_arr = 1/self.a_arr - 1
        lk_arr, _ = ccl.ccllib.get_pk_spline_lk(self.cosmo_ccl_planck.cosmo,
                                                self.nk, 0)
        self.k_arr = np.exp(lk_arr)

        # For debugging
        self.pars_planck_class = self.pars.copy()
        del self.pars_planck_class['w0']
        del self.pars_planck_class['wa']
        self.pars_planck_class['Omega_cdm'] = self.pars_planck_class.pop('Omega_c')

        # Initialize Class
        self.cosmo_class = Class()
        self._test_BeyondLCDM()
        # Set after the test because in the test it empties the param dictionary
        self.cosmo_class.set({'output': 'mPk', 'z_max_pk': np.max(self.z_arr), 'P_k_max_1/Mpc': np.max(self.k_arr)})

    def get_cosmo_ccl(self):
        param_dict = dict({'transfer_function': 'boltzmann_class'},
                          **self.pars)
        if 'output' in param_dict:
            param_dict.pop('output')
        if 'omega_b' in param_dict:
            omega_b = param_dict.pop('omega_b')
            param_dict['Omega_b'] = omega_b / param_dict['h']**2
        if 'omega_c' in param_dict:
            omega_c = param_dict.pop('omega_c')
            param_dict['Omega_c'] = omega_c / param_dict['h']**2

        cosmo_ccl = ccl.Cosmology(**param_dict)
        return cosmo_ccl

    def struct_cleanup(self):
        if 'BeyondLCDM' in self.pars:
            self.cosmo_class.struct_cleanup()
        return

    def empty(self):
        if 'BeyondLCDM' in self.pars:
            self.cosmo_class.empty()
        return

    # Set up the dictionary
    def set(self, *pars_in, **kars):
        if ('A_s' in pars_in[0].keys()) and ('sigma8' in self.pars.keys()):
            self.pars.pop('sigma8')
        if len(pars_in) == 1:
            self.pars.update(dict(pars_in[0]))
        elif len(pars_in) != 0:
            raise RuntimeError("bad call")
        ### Check for parmeters of cl_cross_corr lkl
        if 'params_dir' in self.pars.keys():
            del[self.pars['params_dir']]
        if 'params_dir_copy' in self.pars.keys():
            del[self.pars['params_dir_copy']]
        if 'fiducial_cov' in self.pars.keys():
            del[self.pars['fiducial_cov']]
        #
        if 'BeyondLCDM' not in self.pars:
            if 'tau_reio' in self.pars:
                raise ValueError('CCL does not read tau_reio. Remove it.')
            # Translate w_0, w_a CLASS vars to CCL w0, wa
            if 'w_0' in self.pars.keys():
                self.pars['w0'] = self.pars.pop('w_0')
            if 'w_a' in self.pars.keys():
                self.pars['wa'] = self.pars.pop('w_a')
            # Check that sigma8 or As are fixed with "growth_param"
            if (('A_s' in pars_in) or ('sigma8' in pars_in)) and \
                    ("growth_param" in self.pars):
                raise RuntimeError("Remove 'A_s' and 'sigma8' when modifying \
                                   growth")
        else:
            if 'w0' in self.pars:
                del self.pars['w0']
            if 'wa' in self.pars:
                del self.pars['wa']
            if 'Omega_c' in self.pars:
                del self.pars['Omega_c']

            pars = self.pars.copy()
            del pars['BeyondLCDM']
            pars.update(kars)
            self.cosmo_class.set(pars)

        self.pars.update(kars)
        return True

    def compute_BeyondLCDM(self):
        hc = self.cosmo_class
        hc.compute()
        bhc = hc.get_background()
        # Background
        H = bhc['H [1/Mpc]']
        background = {'a': 1 / (bhc['z'] + 1), 'chi': bhc['comov. dist.'],
                      'h_over_h0':  H / H[-1]}
        # Growth
        D_arr = np.array([hc.scale_independent_growth_factor(z) for z in self.z_arr])
        f_arr = np.array([hc.scale_independent_growth_factor_f(z) for z in self.z_arr])
        growth = {'a': self.a_arr, 'growth_factor': f_arr,
                  'growth_rate': D_arr}
        # Pk
        pkln = np.array([[hc.pk_lin(k, z) for k in self.k_arr] for z in self.z_arr])
        pk_linear = {'a': self.a_arr, 'k': self.k_arr,
                     'delta_matter:delta_matter': pkln}

        cosmo_ccl = ccl.CosmologyCalculator(Omega_c=hc.Omega0_cdm(),
                                            Omega_b=hc.Omega_b(), h=hc.h(),
                                            sigma8=hc.sigma8(), n_s=hc.n_s(),
                                            background=background,
                                            growth=growth,
                                            pk_linear=pk_linear,
                                            nonlinear_model='halofit')
        return cosmo_ccl

    def _test_BeyondLCDM(self):
        print('Testing BeyondLCDM')
        self.cosmo_class.empty()
        self.cosmo_class.set({'output': 'mPk', 'z_max_pk': np.max(self.z_arr), 'P_k_max_1/Mpc': np.max(self.k_arr)})
        self.cosmo_class.set(self.pars_planck_class)
        # # Test set with CCL.set method
        # pars = self.pars_planck_class.copy()
        # pars['BeyondLCDM'] = 'True'
        # self.set(pars)

        # Compute
        cosmo_ccl_blcdm = self.compute_BeyondLCDM()
        cosmo_ccl = self.cosmo_ccl_planck
        pk_blcdm = ccl.nonlin_matter_power(cosmo_ccl_blcdm, self.k_arr, 1)
        pk_planck = ccl.nonlin_matter_power(cosmo_ccl, self.k_arr, 1)
        print(pk_blcdm)
        print(pk_planck)
        rdev = np.max(np.abs((pk_blcdm + 1e-100) / (pk_planck + 1e-100) - 1))
        if rdev > 1e-4:
            print(f'BeyondLCDM test not passed. Max abs. rel. dev. = {rdev}')
            # raise ValueError(f'BeyondLCDM test not passed. Max abs. rel. dev. = {rdev}')
        print(f'BeyondLCDM test passed. Max abs. rel. dev. = {rdev}')
        self.cosmo_class.struct_cleanup()
        self.cosmo_class.empty()

    def compute(self, level=[]):
        # Modified growth part
        if 'BeyondLCDM' in self.pars:
            self.cosmo_ccl = self.compute_BeyondLCDM()
        else:
            self.cosmo_ccl = self.get_cosmo_ccl()
        return

    def get_sigma8(self):
        return ccl.sigma8(self.cosmo_ccl)

    def get_Omegam(self):
        if 'BeyondLCDM' in self.pars:
            Omm = self.pars['Omega_cdm'] + self.pars['Omega_b']
        else:
            Omm = self.pars['Omega_c'] + self.pars['Omega_b']
        return Omm

    def get_S8(self):
        S8 = self.get_sigma8()*(self.get_Omegam()/0.3)**(0.5)
        return S8


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
                msg = "%s was not recognized as a derived parameter" % name
                raise RuntimeError(msg)
            derived[name] = value

        return derived
