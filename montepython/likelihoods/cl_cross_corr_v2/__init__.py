from scipy.interpolate import interp1d
import os
import yaml
import itertools
import numpy as np
from montepython.likelihood_class import Likelihood
import montepython.io_mp as io_mp
import warnings
import pyccl as ccl
import shutil  # To copy the yml file to the outdir
import sacc



class cl_cross_corr_v2(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        ##########
        # First read the YAML file and populate self.use_nuisance
        self.outdir = command_line.folder

        # Read arguments & copy the .yml file to output folder
        with open(os.path.abspath(data.cosmo_arguments['params_dir'])) as f:
            self.params = yaml.safe_load(f)
        shutil.copy2(os.path.abspath(data.cosmo_arguments['params_dir']),
                     self.outdir)

        # TODO: Make this clever
        self.use_nuisance = self.params['use_nuisance']

        # Initialize the Likelihood class
        Likelihood.__init__(self, path, data, command_line)
        ##########

        # Load sacc files
        self.scovG = sacc.Sacc.load_fits(self.params['sacc_covG'])
        if os.path.isfile(self.params['sacc_covNG']):
            scovNG = sacc.Sacc.load_fits(self.params['sacc_covNG'])

        # Check used tracers are in the sacc file
        tracers_sacc = [trd for trd in self.scovG.tracers]
        for tr in self.params['tracers'].keys():
            if tr not in tracers_sacc:
                raise ValueError('The tracer {} is not present in {}'.format(tr, self.params['sacc_covG']))

        # TODO: Remove unused tracers
        # used_tracers = self.params['tracers'].keys()

        # Remove unused Cls
        self.scovG.remove_selection(data_type='cl_eb')
        self.scovG.remove_selection(data_type='cl_be')
        self.scovG.remove_selection(data_type='cl_bb')
        scovNG.remove_selection(data_type='cl_eb')
        scovNG.remove_selection(data_type='cl_be')
        scovNG.remove_selection(data_type='cl_bb')

        # Remove unused tracer combinations
        used_tracer_combinations = []
        for tracers_d in self.params['tracer_combinations']:
            tr1, tr2 = tracers_d['tracers']
            used_tracer_combinations.append((tr1, tr2))
            # Take into account that both are the same when comparing!
            used_tracer_combinations.append((tr2, tr1))

        used_tracer_combinations_sacc = []
        for tracers in self.scovG.get_tracer_combinations():
            if tracers not in used_tracer_combinations:
                self.scovG.remove_selection(tracers=tracers)
                scovNG.remove_selection(tracers=tracers)
            used_tracer_combinations_sacc.append(tracers)

        # Cut scales below and above scale cuts for each tracer combination
        for tracers_d in self.params['tracer_combinations']:
                lmin, lmax = tracers_d['ell_cuts']
                tracers = tuple(tracers_d['tracers'])
                print(tracers, lmin, lmax)
                if tracers not in used_tracer_combinations_sacc:
                    # if not present is because they have the opposite ordering
                    tracers = tracers[::-1]
                self.scovG.remove_selection(ell__lt=lmin, tracers=tracers)
                self.scovG.remove_selection(ell__gt=lmax, tracers=tracers)
                scovNG.remove_selection(ell__lt=lmin, tracers=tracers)
                scovNG.remove_selection(ell__gt=lmax, tracers=tracers)

        # Save the ells considered for each Cl
        ells = np.array([])
        for tr0, tr1 in self.scovG.get_tracer_combinations():
            ells = np.concatenate((ells, self.scovG.get_ell_cl('cl_ee', tr0, tr1)[0]))

        # Save the data vector
        self.data = self.scovG.mean

        # Compute the full covmat
        self.cov = self.scovG.covariance.covmat + scovNG.covariance.covmat

        # Invert covariance matrix
        self.icov = np.linalg.inv(self.cov)

        # Print vector size and dof
        npars = len(data.get_mcmc_parameters(['varying']))
        vecsize = self.data.size
        self.dof = vecsize - npars
        print('    -> Varied parameters = {}'.format(npars))
        print('    -> cl_cross_corr data vector size = {}'.format(vecsize))
        print('    -> cl_cross_corr dof = {}'.format(self.dof))

        # Save a copy of the covmat, ells and cls for the tracer_combinations used for debugging
        np.savez_compressed(os.path.join(self.outdir, 'cl_cross_corr_data_info.npz'), cov=self.cov,
                            ells=ells, cls=self.data, tracers=self.scovG.get_tracer_combinations(), dof=self.dof)
        # end of initialization


    def get_loggaussprior(self, value, center, var):
        lp = -0.5 * ((value - center) / var)**2.
        return lp

    def get_interpolated_cl(self, cosmo, ls, ccl_tracers, tr1, tr2):
        ls_nodes = np.unique(np.geomspace(2, ls[-1], 30).astype(int)).astype(float)
        cls_nodes = ccl.angular_cl(cosmo.cosmo_ccl,
                                   ccl_tracers[tr1],
                                   ccl_tracers[tr2],
                                   ls_nodes)
        cli = interp1d(np.log(ls_nodes), cls_nodes,
                       fill_value=0, bounds_error=False)
        msk = ls >= 2
        cls = np.zeros(len(ls))
        cls[msk] = cli(np.log(ls[msk]))
        return cls

    # compute likelihood

    def loglkl(self, cosmo, data):

        # Initialize logprior
        lp = 0.

        # Initialize dictionary with ccl_tracers
        ccl_tracers = {}

        # Get Tracers
        for trname, trvals in self.params['tracers'].items():
            stracer = self.scovG.get_tracer(trname)
            z = stracer.z
            pz = stracer.nz

            if 'dz' in trvals:
                # Calculate z bias
                pname = '{}_dz_{}'.format(trvals['type'], trvals['bin'])
                dz = data.mcmc_parameters[pname]['current']*data.mcmc_parameters[pname]['scale']
                # Get log prior for dz
                lp = lp + self.get_loggaussprior(dz, *trvals['dz'])
                # Compute z - dz nd remove points with z_dz < 0:
                z_dz = z[z >= dz] - dz
                pz = pz[z >= dz]

            if trvals['type'] == 'gc':
                # Calculate bias
                pname = 'gc_b_{}'.format(trvals['bin'])
                bias = data.mcmc_parameters[pname]['current']*data.mcmc_parameters[pname]['scale']
                bz = bias*np.ones(z.shape)
                # Get tracer
                ccl_tracers[trname] = ccl.NumberCountsTracer(cosmo.cosmo_ccl,has_rsd=False,dndz=(z_dz, pz),bias=(z, bz))
            elif trvals['type'] == 'wl':
                # Get log prior for m
                pname = 'wl_m_{}'.format(trvals['bin'])
                value = data.mcmc_parameters[pname]['current']*data.mcmc_parameters[pname]['scale']
                lp = lp + self.get_loggaussprior(value, *trvals['m'])
                # Calculate bias IA
                A = data.mcmc_parameters['wl_ia_A']['current']*data.mcmc_parameters['wl_ia_A']['scale']
                eta = data.mcmc_parameters['wl_ia_eta']['current']*data.mcmc_parameters['wl_ia_eta']['scale']
                z0 = data.mcmc_parameters['wl_ia_z0']['current']*data.mcmc_parameters['wl_ia_z0']['scale']
                bz = A*((1.+z)/(1.+z0))**eta*0.0139/0.013872474  # pyccl2 -> has already the factor inside. Only needed bz
                # Get tracer
                ccl_tracers[trname] = ccl.WeakLensingTracer(cosmo.cosmo_ccl, dndz=(z_dz,pz), ia_bias=(z,bz))
            elif trvals['type'] == 'cv':
                ccl_tracers[trname] = ccl.CMBLensingTracer(cosmo.cosmo_ccl, z_source=1100)#TODO: correct z_source
            else:
                raise ValueError('Type of tracer not recognized. It can be gc, wl or cv!')

        # Get theory Cls

        theory = np.zeros_like(self.data)
        for tr1, tr2 in self.scovG.get_tracer_combinations():
            # Get the indices for this part of the data vector
            ind = self.scovG.indices(data_type='cl_ee', tracers=(tr1, tr2))
            # Get the bandpower window function.
            # w.values contains the values of ell at which it is sampled
            w = self.scovG.get_bandpower_windows(ind)
            # Unbinned power spectrum.
            if self.params['interpolate_cls'] is True:
                cl_unbinned = self.get_interpolated_cl(cosmo, w.values, ccl_tracers, tr1, tr2)
            else:
                cl_unbinned = ccl.angular_cl(cosmo.cosmo_ccl, ccl_tracers[tr1], ccl_tracers[tr2], w.values)
            # Convolved with window functions.
            cl_binned = np.dot(w.weight.T, cl_unbinned)
            for tr in [tr1, tr2]:
                trvals = self.params['tracers'][tr]
                if  trvals['type'] == 'wl':
                    pname = 'wl_m_{}'.format(trvals['bin'])
                    m = data.mcmc_parameters[pname]['current']*data.mcmc_parameters[pname]['scale']
                    cl_binned = (1.+m)*cl_binned

            # Assign to theory vector.
            theory[ind] = cl_binned

        # Get chi2
        chi2 = (self.data-theory).dot(self.icov).dot(self.data-theory)

        lkl = lp - 0.5 * chi2

        # np.savez_compressed(os.path.join(self.outdir, 'cl_cross_corr_bestfit_info.npz'), chi2_nolp=chi2, lp_chi2=2*lp, chi2=2*lkl, chi2dof=2*lkl/self.dof,
        #                     cls=theory, ells=self.ells_tosave, tracers=self.tracers_tosave)


        return lkl
