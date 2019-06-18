import os
import yaml
import itertools
import numpy as np
from montepython.likelihood_class import Likelihood
import montepython.io_mp as io_mp
import warnings
import pyccl as ccl



class cl_cross_corr(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)


        def find_file_cls(path, tracers):
            for p in itertools.permutations(tracers):
                fname = '{}/cls_{}_{}.npz'.format(path,p[0],p[1])
                if os.path.isfile(fname):
                    return fname
            raise IOError('File does not exists')


        def find_file_cov(path, tracers1, tracers2):
            for p_ref in itertools.permutations([tracers1,tracers2]):
                for p1 in itertools.permutations(p_ref[0]):
                    for p2 in itertools.permutations(p_ref[1]):
                        fname = '{}/cov_{}_{}_{}_{}.npz'.format(path,p1[0],p1[1],p2[0],p2[1])
                        if os.path.isfile(fname):
                            return fname
            raise IOError('File does not exists')


        # Read arguments
        with open(os.path.abspath(data.cosmo_arguments['params_dir'])) as f:
            self.params = yaml.safe_load(f)
        self.n_data_vectors = len(self.params['data_vectors'])


        # Load Cl's
        self.data = np.array([])
        used_tracers = np.array([])
        for ndv in range(self.n_data_vectors):
            dv = self.params['data_vectors'][ndv]
            # Find ell-cuts
            ell_cuts_1 = next(x['ell_cuts'] for x in self.params['maps'] if x['name']==dv['tracers'][0])
            ell_cuts_2 = next(x['ell_cuts'] for x in self.params['maps'] if x['name']==dv['tracers'][1])
            dv['ell_cuts'] = [max(ell_cuts_1[0],ell_cuts_2[0]),min(ell_cuts_1[1],ell_cuts_2[1])]
            # Get ells and cls
            fname = find_file_cls(self.cov_cls,dv['tracers'])
            ells = np.load(fname)['ells']
            cls = np.load(fname)['cls']
            dv['mask'] = (ells>=dv['ell_cuts'][0]) & (ells<=dv['ell_cuts'][1])
            dv['ells'] = ells[dv['mask']]
            self.data = np.append(self.data,cls[dv['mask']])
            used_tracers = np.append(used_tracers,dv['tracers'])
        used_tracers = np.unique(used_tracers)

        # Remove unused tracers
        for ntr, tr in enumerate(self.params['maps']):
            if tr['name'] not in used_tracers:
                self.params['maps'].pop(ntr)

        # Load dndz
        for tr in self.params['maps']:
            if tr['type'] in ['gc', 'wl']:
                fname = self.dndz+'/'+tr['dndz_file']
                tr['dndz'] = np.loadtxt(fname,unpack=True)

        # Load Covmat
        self.cov = np.zeros((len(self.data),len(self.data)))
        for ndv1 in range(self.n_data_vectors):
            dv1 = self.params['data_vectors'][ndv1]
            s1 = int(np.array([self.params['data_vectors'][x]['mask'] for x in range(ndv1)]).sum())
            e1 = s1+dv1['mask'].sum()
            for ndv2 in range(ndv1,self.n_data_vectors):
                dv2 = self.params['data_vectors'][ndv2]
                s2 = int(np.array([self.params['data_vectors'][x]['mask'] for x in range(ndv2)]).sum())
                e2 = s2+dv2['mask'].sum()
                fname = find_file_cov(self.cov_cls,dv1['tracers'],dv2['tracers'])
                cov = np.load(fname)['arr_0']
                cov = cov[dv1['mask'],:][:,dv2['mask']]
                self.cov[s1:e1,s2:e2] = cov
                self.cov[s2:e2,s1:e1] = cov.T

        # Invert covariance matrix
        self.icov = np.linalg.inv(self.cov)

        # end of initialization



    # compute likelihood

    def loglkl(self, cosmo, data):

        # Get Tracers
        for tr in self.params['maps']:
            if tr['type'] == 'gc':
                # Import z, pz
                z  = tr['dndz'][1]
                pz = tr['dndz'][3]
                # Calculate z bias
                pname = 'gc_dz_{}'.format(tr['bin'])
                dz = data.mcmc_parameters[pname]['current']*data.mcmc_parameters[pname]['scale']
                z_dz = z-dz
                # Calculate bias
                pname = 'gc_b_{}'.format(tr['bin'])
                bias = data.mcmc_parameters[pname]['current']*data.mcmc_parameters[pname]['scale']
                bz = bias*np.ones(z.shape)
                # Get tracer
                tr['tracer'] = ccl.NumberCountsTracer(cosmo.cosmo_ccl,has_rsd=False,dndz=(z_dz,pz),bias=(z,bz))
            elif tr['type'] == 'wl':
                # Import z, pz
                z  = tr['dndz'][1]
                pz = tr['dndz'][3]
                # Calculate z bias
                pname = 'wl_dz_{}'.format(tr['bin'])
                dz = data.mcmc_parameters[pname]['current']*data.mcmc_parameters[pname]['scale']
                z_dz = z-dz
                # Calculate bias IA
                A = data.mcmc_parameters['wl_ia_A']['current']*data.mcmc_parameters['wl_ia_A']['scale']
                eta = data.mcmc_parameters['wl_ia_eta']['current']*data.mcmc_parameters['wl_ia_eta']['scale']
                z0 = data.mcmc_parameters['wl_ia_z0']['current']*data.mcmc_parameters['wl_ia_z0']['scale']
                Omm = cosmo.get_Omegam()
                Dz = ccl.background.growth_factor(cosmo.cosmo_ccl, 1./(1.+z))
                bz = A*((1.+z)/(1.+z0))**eta*0.0139*Omm/Dz #TODO: ask David about Eq 10 in 1810.02322
                fz = np.ones(z.shape)
                # Get tracer
                tr['tracer'] = ccl.WeakLensingTracer(cosmo.cosmo_ccl,dndz=(z_dz,pz),ia_bias=(z,bz),red_frac=(z,fz))
            elif tr['type'] == 'cv':
                tr['tracer'] = ccl.CMBLensingTracer(cosmo.cosmo_ccl, z_source=1100)#TODO: correct z_source
            else:
                raise ValueError('Type of tracer not recognized. It can be gc, wl or cv!')

        # Get theory Cls
        theory = np.array([])
        for ndv in range(self.n_data_vectors):
            dv = self.params['data_vectors'][ndv]
            tracer1 = next(x['tracer'] for x in self.params['maps'] if x['name']==dv['tracers'][0])
            tracer2 = next(x['tracer'] for x in self.params['maps'] if x['name']==dv['tracers'][1])
            cls = ccl.angular_cl(cosmo.cosmo_ccl, tracer1, tracer2, dv['ells'])
            # Add multiplicative bias to WL
            type1 = next(x['type'] for x in self.params['maps'] if x['name']==dv['tracers'][0])
            type2 = next(x['type'] for x in self.params['maps'] if x['name']==dv['tracers'][1])
            if type1 == 'wl':
                bin = next(x['bin'] for x in self.params['maps'] if x['name']==dv['tracers'][0])
                pname = 'wl_m_{}'.format(bin)
                m = data.mcmc_parameters[pname]['current']*data.mcmc_parameters[pname]['scale']
                cls = (1.+m)*cls
            if type2 == 'wl':
                bin = next(x['bin'] for x in self.params['maps'] if x['name']==dv['tracers'][1])
                pname = 'wl_m_{}'.format(bin)
                m = data.mcmc_parameters[pname]['current']*data.mcmc_parameters[pname]['scale']
                cls = (1.+m)*cls
            theory = np.append(theory,cls)

        # Get chi2
        print(len(theory))
        chi2 = (self.data-theory).dot(self.icov).dot(self.data-theory)
        lkl = - 0.5 * chi2

        return lkl
