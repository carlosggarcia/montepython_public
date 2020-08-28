"""
.. module:: recompute
    :synopsis: Recompute existing chains

.. moduleauthor:: Benjamin Audren <benjamin.audren@epfl.ch>
"""
import os
import sys
import io_mp
import sampler
from data import Data


def run(cosmo, data, command_line):
    """
    Recompute existing chains.
    """
    starting_folder = command_line.Rec_folder
    if not starting_folder:
        raise io_mp.ConfigurationError(
            "You must specify a folder or a set of chains with the option \
            '--Rec-folder'")
    chains = []
    # If starting_folder is of length 1, it means it is either a whole folder,
    # or just one chain. If it is a folder, we recover all chains within.
    if (len(starting_folder) == 1) and (os.path.isdir(starting_folder[0])):
        starting_folder = starting_folder[0]
        for elem in os.listdir(starting_folder):
            if elem.find("__") != -1:
                chains.append(elem)
    # Else, it is a list of chains, of which we recover folder name, and store
    # all of them in chains.
    else:
        chains = starting_folder
        starting_folder = os.path.sep.join(chains[0].split(os.path.sep)[:-1])
        chains = [elem.split(os.path.sep)[-1] for elem in chains]

    for elem in chains:
        translate_chain(data, cosmo, command_line, starting_folder, elem)


def translate_chain(data, cosmo, command_line,
                    starting_folder, chain_name):
    """
    Translate the input to the output
    """

    input_path = os.path.join(starting_folder, chain_name)
    output_path = os.path.join(command_line.folder, chain_name)
    print(' -> reading ', input_path)
    parameter_names = data.get_mcmc_parameters(['varying'])
    with open(input_path, 'r') as input_chain:
        with open(output_path, 'w') as output_chain:
            for line in input_chain:
                # T. Brinckmann: Added next 3 lines for compatibility with --update
                if line[0]=='#':
                    output_chain.write(line)
                    continue
                params = line.split()
                # recover the likelihood of this point
                loglike = -float(params[1])
                # Assign all the recovered values to the data structure
                for index, param in enumerate(parameter_names):
                    data.mcmc_parameters[param]['current'] = \
                        float(params[2+index])
                data.update_cosmo_arguments()

                newloglike = sampler.compute_lkl(cosmo, data)

                # Accept the point
                sampler.accept_step(data)
                io_mp.print_vector([output_chain], 0, newloglike, data)
                sys.stdout.write("loglike: {} -> {}\n".format(loglike, newloglike))
    print(output_path, 'written')


