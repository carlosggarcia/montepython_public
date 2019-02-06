import re
import random
import numpy as np
import io_mp

class CCL():
    """
    General class for the CCL object.

    """

    def __init__(self):

        try:
            import pyccl as ccl
        except ImportError:
            raise io_mp.MissingLibraryError(
                "You must have installed the pyccl module.")

        self.state = 1

        # Set default parameters
        self.pars = {
        }


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
        """
        get_current_derived_parameters(names)

        Return a dictionary containing an entry for all the names defined in the
        input list.

        Parameters
        ----------
        names : list
                Derived parameters that can be asked from Monte Python, or
                elsewhere.

        Returns
        -------
        derived : dict

        .. warning::

            This method used to take as an argument directly the data class from
            Monte Python. To maintain compatibility with this old feature, a
            check is performed to verify that names is indeed a list. If not, it
            returns a TypeError. The old version of this function, when asked
            with the new argument, will raise an AttributeError.

        """
        if type(names) != type([]):
            raise TypeError("Deprecated")

        derived = {}
        for name in names:
            if name == 'sigma8':#TODO:change this
                value = self.state
            else:
                raise RuntimeError("%s was not recognized as a derived parameter" % name)
            derived[name] = value
        return derived

        return derived
