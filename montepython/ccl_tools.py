import re
import random
import numpy as np


def read_bool(value):
    if re.match('y.+', value, re.IGNORECASE):
        return True
    elif re.match('n.+', value, re.IGNORECASE):
        return False
    else:
        raise IOError('Boolean type not recognized!')


def how_many_sims(n_sims, n_sims_tot, n_data, n_data_tot):
    # If all simulations wanted, return it
    if n_sims=='all':
        return n_sims_tot

    # If auto, calculate how many sims should be used
    elif n_sims=='auto':
        ratio = (n_sims_tot-n_data_tot-2.)/(n_sims_tot-1.)
        return int(round((2.+n_data-ratio)/(1.-ratio)))

    # If it is a number, just return it
    else:
        return int(n_sims)


def clean_cl(cl, noise):
    if cl.ndim==4:
        return cl - noise
    elif cl.ndim==5:
        clean = np.array([cl[:,x]-noise for x in range(len(cl[0]))])
        clean = np.transpose(clean,axes=(1,0,2,3,4))
        return clean
    else:
        raise ValueError('Expected Cl\'s array with dimensions 4 or 5. Found {}'.format(cl.ndim))


def apply_kl(kl_t, corr, method, scale_dep, n_kl):
    """ Apply the KL transform to the correlation function
        and reduce number of dimensions.

    Args:
        kl_t: KL transform.
        corr: correlation function

    Returns:
        KL transformed correlation function.

    """

    kl_t_T = np.moveaxis(kl_t,[-1],[-2])

    # Apply KL transform
    corr_kl = np.dot(kl_t,corr)
    if scale_dep:
        corr_kl = np.diagonal(corr_kl,axis1=0,axis2=-2)
        corr_kl = np.moveaxis(corr_kl,[-1],[-2])
    corr_kl = np.dot(corr_kl,kl_t_T)
    if scale_dep:
        corr_kl = np.diagonal(corr_kl,axis1=-3,axis2=-2)
        corr_kl = np.moveaxis(corr_kl,[-1],[-2])
    corr_kl = np.moveaxis(corr_kl,[0],[-2])

    # Reduce dimensions of the array
    corr_kl = np.moveaxis(corr_kl,[-2,-1],[0,1])
    corr_kl = corr_kl[:n_kl,:n_kl]
    corr_kl = np.moveaxis(corr_kl,[0,1],[-2,-1])
    if method == 'kl_diag':
        corr_kl = np.diagonal(corr_kl,  axis1=-2, axis2=-1)

    return corr_kl


def select_sims(sims, n_sims):

    # Select simulations
    rnd = random.sample(range(sims.shape[1]), n_sims)
    less_sims = sims[:,rnd]

    return less_sims


def flatten_cl(cl, is_diag):
    flat_cl = cl
    if not is_diag:
        tr_idx = np.triu_indices(cl.shape[-1])
        flat_cl = np.moveaxis(flat_cl,[-2,-1],[0,1])
        flat_cl = flat_cl[tr_idx]
        flat_cl = np.moveaxis(flat_cl,[0],[-1])
    flat_cl = flat_cl.reshape(flat_cl.shape[:-2]+(flat_cl.shape[-2]*flat_cl.shape[-1],))
    return flat_cl


def unflatten_cl(cl, shape, is_diag):
    if is_diag:
        unflat_cl = cl.reshape(shape)
    else:
        tr_idx = np.triu_indices(shape[-1])
        unflat_cl = np.zeros(shape)
        tmp_cl = cl.reshape(shape[:-2]+(-1,))
        tmp_cl = np.moveaxis(tmp_cl,[-1],[0])
        unflat_cl = np.moveaxis(unflat_cl,[-2,-1],[0,1])
        unflat_cl[tr_idx] = tmp_cl
        unflat_cl = np.moveaxis(unflat_cl,[1],[0])
        unflat_cl[tr_idx] = tmp_cl
        unflat_cl = np.moveaxis(unflat_cl,[0,1],[-2,-1])
    return unflat_cl


def flatten_covmat(cov, is_diag):
    if is_diag:
        flat_cov = np.moveaxis(cov,[-3,-2],[-2,-3])
        idx = 2
    else:
        flat_cov = np.moveaxis(cov,[-5,-4,-3,-2],[-3,-5,-2,-4])
        idx = 3
    flat_cov = flatten_cl(flat_cov, is_diag)
    flat_cov = np.moveaxis(flat_cov,[-1],[-1-idx])
    flat_cov = flatten_cl(flat_cov, is_diag)
    return flat_cov


def unflatten_covmat(cov, cl_shape, is_diag):
    unflat_cov = np.apply_along_axis(unflatten_cl, -1, cov, cl_shape, is_diag)
    unflat_cov = np.apply_along_axis(unflatten_cl, -1-len(cl_shape), unflat_cov, cl_shape, is_diag)
    if is_diag:
        unflat_cov = np.moveaxis(unflat_cov,[-3,-2],[-2,-3])
    else:
        unflat_cov = np.moveaxis(unflat_cov,[-5,-4,-3,-2],[-4,-2,-5,-3])
    return unflat_cov


def get_covmat(sims, is_diag):
    sims_flat = flatten_cl(sims, is_diag)
    if len(sims_flat.shape) == 2:
        cov = np.cov(sims_flat.T,bias=True)
    elif len(sims_flat.shape) == 3:
        cov = np.array([np.cov(x.T,bias=True) for x in sims_flat])
    else:
        raise ValueError('Input dimensions can be either 2 or 3, found {}'.format(len(sims_flat.shape)))
    if is_diag:
        shape = sims.shape[-2:]
    else:
        shape = sims.shape[-3:]
    return unflatten_covmat(cov, shape, is_diag)


def unify_fields_cl(cl, cov_pf, is_diag):
    cl_flat = flatten_cl(cl, is_diag)
    cov = flatten_covmat(cov_pf, is_diag)
    inv_cov = np.array([np.linalg.inv(x) for x in cov])
    tot_inv_cov = np.sum(inv_cov,axis=0)
    tot_cl = np.array([np.linalg.solve(cov[x], cl_flat[x].T) for x in range(len(cl))])
    tot_cl = np.sum(tot_cl, axis=0)
    tot_cl = np.linalg.solve(tot_inv_cov, tot_cl).T
    tot_cl = unflatten_cl(tot_cl, cl.shape[1:], is_diag)
    return tot_cl


def mask_cl(cl, mask, is_diag):
    if is_diag:
        idx = -2
    else:
        idx = -3
    mask_cl = np.moveaxis(cl,[idx],[0])
    mask_cl = mask_cl[mask]
    mask_cl = np.moveaxis(mask_cl,[0],[idx])
    return mask_cl
