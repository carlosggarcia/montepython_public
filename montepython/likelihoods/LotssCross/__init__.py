import yaml
import itertools
import numpy as np
from montepython.likelihood_class import Likelihood
import pyccl as ccl


class LotssCross(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        def find_cls(data, prefix, tr1):
            for p in itertools.permutations(tr1):
                try:
                    return data[prefix+''.join(p)]
                except KeyError:
                    pass
            raise KeyError('Cls do not exist')

        def find_cov(data, tr1, tr2):
            for p_ref in itertools.permutations([tr1, tr2]):
                for p1 in itertools.permutations(p_ref[0]):
                    for p2 in itertools.permutations(p_ref[1]):
                        case = [x for y in [p1, p2] for x in y]
                        try:
                            return data['cov_'+''.join(case)]
                        except KeyError:
                            pass
            raise KeyError('Covariance matrix does not exist')

        # Read ell cuts
        try:
            with open(data.arguments['maps']) as f:
                self.maps = yaml.safe_load(f)
        except KeyError:
            self.maps = self.def_maps.copy()

        # Read data
        try:
            data_all = np.load(data.arguments['cl_cov_nz'])
        except KeyError:
            data_all = np.load(data.arguments['def_cl_cov_nz'])
        ells = data_all['l_eff']

        # Load Cl's
        self.cl_data = np.array([])
        used_tracers = np.array([])
        for dv in self.maps['data_vectors']:
            # Type of tracers
            tp1 = next(x['type'] for x in self.maps['maps']
                       if x['name'] == dv['tracers'][0])
            tp2 = next(x['type'] for x in self.maps['maps']
                       if x['name'] == dv['tracers'][1])
            dv['types'] = [tp1, tp2]
            # Find ell-cuts
            ell_cuts_1 = next(x['ell_cuts'] for x in self.maps['maps']
                              if x['name'] == dv['tracers'][0])
            ell_cuts_2 = next(x['ell_cuts'] for x in self.maps['maps']
                              if x['name'] == dv['tracers'][1])
            dv['ell_cuts'] = [max(ell_cuts_1[0], ell_cuts_2[0]),
                              min(ell_cuts_1[1], ell_cuts_2[1])]
            # Get ells and cls
            dv['mask'] = (ells >= dv['ell_cuts'][0]) & (
                ells <= dv['ell_cuts'][1])
            dv['ells'] = ells[dv['mask']]
            cl_data = find_cls(data_all, 'cl_', dv['types'])
            cl_noise = find_cls(data_all, 'nl_', dv['types'])
            cl_clean = cl_data[dv['mask']] - cl_noise[dv['mask']]
            self.cl_data = np.append(self.cl_data, cl_clean)
            used_tracers = np.append(used_tracers, dv['tracers'])
        used_tracers = np.unique(used_tracers)

        # Remove unused tracers
        for ntr, tr in enumerate(self.maps['maps']):
            if tr['name'] not in used_tracers:
                self.maps['maps'].pop(ntr)

        # Load dndz
        self.z_g = data_all['z_g']
        self.nz_g = data_all['nz_g']

        # Load Covmat
        self.cov = np.zeros((len(self.cl_data), len(self.cl_data)))
        for ndv1, dv1 in enumerate(self.maps['data_vectors']):
            s1 = int(np.array([self.maps['data_vectors'][x]['mask'] for x in
                              range(ndv1)]).sum())
            e1 = s1+dv1['mask'].sum()
            for ndv2, dv2 in enumerate(self.maps['data_vectors']):
                s2 = int(np.array([self.maps['data_vectors'][x]['mask']
                                   for x in range(ndv2)]).sum())
                e2 = s2+dv2['mask'].sum()
                cov = find_cov(data_all, dv1['types'], dv2['types'])
                cov = cov[dv1['mask'], :][:, dv2['mask']]
                self.cov[s1:e1, s2:e2] = cov
                self.cov[s2:e2, s1:e1] = cov.T

        # Invert covariance matrix
        self.icov = np.linalg.inv(self.cov)

        # end of initialization

    # compute likelihood

    def loglkl(self, cosmo, data):

        # Get Tracers
        for tr in self.maps['maps']:
            if tr['type'] == 'g':
                # Bias
                bz_g = data.mcmc_parameters['bias_g']['current']
                bz_g *= data.mcmc_parameters['bias_g']['scale']
                if data.arguments['bias_type'] == 'one_over_growth':
                    bz_g /= cosmo.get_growth_factor(1./(1.+self.z_g))
                elif data.arguments['bias_type'] == 'constant':
                    bz_g *= np.ones_like(self.z_g)
                else:
                    raise ValueError('Bias type not recognized!')
                # Get tracer
                tr['tracer'] = ccl.NumberCountsTracer(cosmo.cosmo_ccl, False,
                                                      (self.z_g, self.nz_g),
                                                      (self.z_g, bz_g))
            elif tr['type'] == 'k':
                tr['tracer'] = ccl.CMBLensingTracer(cosmo.cosmo_ccl,
                                                    z_source=1100)
            else:
                raise ValueError('Type of tracer can be g, or k!')

        # Get theory Cls
        cl_theory = np.array([])
        for dv in self.maps['data_vectors']:
            tracer1 = next(x['tracer'] for x in self.maps['maps']
                           if x['name'] == dv['tracers'][0])
            tracer2 = next(x['tracer'] for x in self.maps['maps']
                           if x['name'] == dv['tracers'][1])
            cls = ccl.angular_cl(cosmo.cosmo_ccl, tracer1, tracer2, dv['ells'])
            cl_theory = np.append(cl_theory, cls)

        # Get chi2
        chi2 = (self.cl_data-cl_theory).dot(self.icov)
        chi2 = chi2.dot(self.cl_data-cl_theory)
        # Contours:
        # sigma8 vs bias: 2 scenarios (bias constant, bias = c/delta)
        # only auto-g and auto-g + cross-gl

        lkl = - 0.5 * chi2

        return lkl
