import numpy as np
import pyccl as ccl


class CCL():
    """
    General class for the CCL object.

    """

    def __init__(self):

        self.state = 1

        # Set default parameters
        self.pars = {
            'h':       0.67,
            'Omega_c': 0.27,
            'Omega_b': 0.045,
            'sigma8': 0.840,
            'n_s':     0.96,
            'w0': -1.0,
            'wa':  0.0
        }

    def get_cosmo_ccl(self):
        param_dict = dict({'transfer_function': 'boltzmann_class'},
                          **self.pars)
        try:
            param_dict.pop('output')
        except KeyError:
            pass
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

    def struct_cleanup(self):
        return

    def empty(self):
        return

    # Set up the dictionary
    def set(self, *pars_in, **kars):
        if 'A_s' in pars_in[0].keys():
            self.pars.pop('sigma8')
        if len(pars_in) == 1:
            self.pars.update(dict(pars_in[0]))
        elif len(pars_in) != 0:
            raise RuntimeError("bad call")
        self.pars.update(kars)
        return True

    def compute(self, level=[]):
        self.cosmo_ccl = self.get_cosmo_ccl()
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
            else:
                msg = "%s was not recognized as a derived parameter" % name
                raise RuntimeError(msg)
            derived[name] = value

        return derived

    def dpk(self, a):
        result = 0
        if self.pars['growth_param'] == 'linear':
            i = 0
            while True:
                pname = 'dpk' + str(i)
                if pname not in self.pars:
                    break
                dpki = self.pars[pname]
                result += dpki / np.math.factorial(i) * (1-a)**i
                i += 1
        return result

    def pk2D_new(self, pk):
        def pknew(k, a):
            return (1 + self.dpk(a)) ** 2 * pk.eval(k, a, self.cosmo_ccl)
        return pknew
