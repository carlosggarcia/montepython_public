import os
import numpy as np
from montepython.likelihood_class import Likelihood
import montepython.io_mp as io_mp
import warnings


class growth(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # define array for values of z and data points
        self.z = np.array([], 'float64')
        self.data = np.array([], 'float64')
        self.error = np.array([], 'float64')

        # read redshifts and data points
        with open(os.path.join(self.data_directory, self.file), 'r') as filein:
            for line in filein:
                if line.find('#') == -1:
                    # the first entry of the line is the identifier
                    this_line = line.split()
                    self.z = np.append(self.z, float(this_line[1]))
                    self.data = np.append(self.data, float(this_line[2]))
                    self.error = np.append(self.error, float(this_line[3]))

        # number of data points
        self.num_points = np.shape(self.z)[0]

        # end of initialization

    # compute likelihood

    def loglkl(self, cosmo, data):

        chi2 = 0.

        # for each point, compute growth rate f, power spectrum normalization sig8,
        # theoretical prediction and chi2 contribution

        for i in range(self.num_points):

            s8 = cosmo.sigma8_at_z(self.z[i])
            f  = cosmo.growthrate_at_z(self.z[i])
            theo = f*s8

            chi2 += ((theo - self.data[i]) / self.error[i]) ** 2

        # return ln(L)
        lkl = - 0.5 * chi2
        return lkl
