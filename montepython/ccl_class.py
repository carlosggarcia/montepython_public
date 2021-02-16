import numpy as np
import pyccl as ccl
from scipy.interpolate import interp1d


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

        self.cosmo_ccl_planck = self.get_cosmo_ccl()

    def get_cosmo_ccl(self):
        param_dict = dict({'transfer_function': 'boltzmann_class'},
                          **self.pars)
        try:
            param_dict.pop('output')
        except KeyError:
            pass
        if 'growth_param' in param_dict:
            param_dict.pop('growth_param')
            for k in list(param_dict.keys()):
                if 'dpk' in k:
                    param_dict.pop(k)

        cosmo_ccl = ccl.Cosmology(**param_dict)
        return cosmo_ccl

    def get_sigma8(self):
        return ccl.sigma8(self.cosmo_ccl)

    def get_Omegam(self):
        Omm = self.pars['Omega_c'] + self.pars['Omega_b']
        return Omm

    def get_S8(self):
        S8 = self.get_sigma8()*(self.get_Omegam()/0.3)**(0.5)
        return S8

    def get_growth_factor(self, a):
        return ccl.background.growth_factor(self.cosmo_ccl, a)

    def struct_cleanup(self):
        return

    def empty(self):
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
        if 'fiducial_cov' in self.pars.keys():
            del[self.pars['fiducial_cov']]
        #
        if 'tau_reio' in self.pars.keys():
            raise ValueError('CCL does not read tau_reio. Remove it.')
        # Translate w_0, w_a CLASS vars to CCL w0, wa
        if 'w_0' in self.pars.keys():
            self.pars['w0'] = self.pars.pop('w_0')
        if 'w_a' in self.pars.keys():
            self.pars['wa'] = self.pars.pop('w_a')

        self.pars.update(kars)
        return True

    def compute(self, level=[]):
        try:
            self.cosmo_ccl = self.get_cosmo_ccl()
        except:
            print('Error for', self.pars)
        # Modified growth part
        if 'growth_param' in self.pars:
            pk = ccl.boltzmann.get_class_pk_lin(self.cosmo_ccl)
            pknew = ccl.Pk2D(pkfunc=self.pk2D_new(pk), cosmo=self.cosmo_ccl,
                             is_logp=False)
            ccl.ccllib.cosmology_compute_linear_power(self.cosmo_ccl.cosmo,
                                                      pknew.psp, 0)

        # self.cosmo_ccl.compute_nonlin_power()
        ccl.sigma8(self.cosmo_ccl)  # David's suggestion
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
            elif 'S8z_' in name:
                z = float(name.split('_')[-1])
                value = self.get_S8z(z)
            elif 'sigma8z_' in name:
                z = float(name.split('_')[-1])
                value = self.get_sigma8z(z)
            else:
                msg = "%s was not recognized as a derived parameter" % name
                raise RuntimeError(msg)
            derived[name] = value

        return derived

    def get_S8z(self, z):
        Omega_m = self.get_Omegam()
        sigma8z = self.get_sigma8z(z)
        S8z = sigma8z * (Omega_m/0.3)**(0.5)
        return S8z

    def get_sigma8z(self, z):
        a = 1 / (1 + z)
        D_new = self.get_D_new(a)
        return D_new **2. *  self.get_sigma8()

    def get_D_new(self, a):
        if self.pars['growth_param'] == 'taylor':
            # D(z) = (dpk0 + dpk1 * (1 - a) + ... ) * D_Planck(z)
            result = 0
            i = 0
            while True:
                pname = 'dpk' + str(i)
                if pname not in self.pars:
                    break
                dpki = self.pars[pname]
                result += dpki / np.math.factorial(i) * (1-a)**i
                i += 1
            result *= ccl.growth_factor(self.cosmo_ccl_planck, a)
        elif self.pars['growth_param'] == 'binning':
            # D(z) = D_binned(z)
            z_Dz = []
            for pname, pvalue in self.pars.items():
                if 'dpk' in pname:
                    z = float(pname.split('_')[-1])
                    z_Dz.append((z, pvalue))

            z_Dz = np.array(sorted(z_Dz)).T

            result = interp1d(z_Dz[0], z_Dz[1], kind='cubic',
                              fill_value='extrapolate', assume_sorted=True)
        else:
            raise ValueError('growth_param {self.pars["growth_param"]} not implemented.')

        return result

    def pk2D_new(self, pk):
        def pknew(k, a):
            D_new = self.get_D_new(a)
            return D_new ** 2 * pk.eval(k, 1, self.cosmo_ccl)
        return pknew
