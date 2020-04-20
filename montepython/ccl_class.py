import re
import random
import numpy as np
import io_mp
import pyccl as ccl



class CCL():
    """
    General class for the CCL object.

    """

    def __init__(self):

        self.state = 1

        # Set default parameters
        self.pars = {
            'h'               :  0.67,
            'Omega_c'         :  0.27,
            'Omega_b'         :  0.045,
            'A_s'             :  2.8,
            # 'ln10_A_s'        :  3.044522,
            # 'sigma_8'          :  0.840421163375,
            'n_s'             :  0.96,
            'w_0'             : -1.0,
            'w_a'             :  0.0
        }


    def get_cosmo_ccl(self):
        cosmo_ccl = ccl.Cosmology(
            h        = self.pars['h'],
            Omega_c  = self.pars['Omega_c'],
            Omega_b  = self.pars['Omega_b'],
            # sigma8   = self.pars['sigma_8'],
            # A_s      = (10.**(-10.))*np.exp(self.pars['ln10_A_s']),
            A_s      = self.pars['A_s'],
            n_s      = self.pars['n_s'],
            w0       = self.pars['w_0'],
            wa       = self.pars['w_a'],
            transfer_function = 'boltzmann_class'
        )
        return cosmo_ccl

    def get_sigma8(self):
        return ccl.sigma8(self.cosmo_ccl)


    def get_Omegam(self):
        Omm = self.pars['Omega_c']+self.pars['Omega_b']
        return Omm


    def get_S8(self):
        S8 = self.get_sigma8()*(self.get_Omegam()/0.3)**(0.5)
        return S8

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
        self.cosmo_ccl = self.get_cosmo_ccl()
        #### Modified growth part
        if 'growth_param' in self.pars:
            pk = ccl.boltzmann.get_class_pk_lin(self.cosmo_ccl)
            pknew = ccl.Pk2D(pkfunc=self.pk2D_new(pk), cosmo=self.cosmo_ccl, is_logp=False)
            ccl.ccllib.cosmology_compute_linear_power(self.cosmo_ccl.cosmo, pknew.psp, 0)

        #self.cosmo_ccl.compute_nonlin_power()
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
                raise RuntimeError("%s was not recognized as a derived parameter" % name)
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

