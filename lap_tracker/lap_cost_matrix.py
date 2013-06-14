#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import numpy.ma as ma
from scipy.spatial.distance import cdist

def get_lap_args(lapmat):

    idxs_in, idxs_out = np.mgrid[:lapmat.shape[0], 
                                 :lapmat.shape[1]]
    idxs_in = idxs_in.flatten()
    idxs_out = idxs_out.flatten()

    flatmat = lapmat.flatten()
    finite_flat = np.isfinite(flatmat)
    if not any(finite_flat):
        print('Warning, no finite element in the LAP matrix')
    costs = flatmat[finite_flat]
    idxs_in = idxs_in[finite_flat]
    idxs_out = idxs_out[finite_flat]
    return idxs_in, idxs_out, costs
    

def get_lapmat(pos0, pos1, max_disp=1000.):

    pos0 = np.asarray(pos0)
    pos1 = np.asarray(pos1)
    num_in, ndim = pos0.shape
    num_out, ndim = pos1.shape
    if ndim not in (2, 3):
        raise ValueError('''Only 2d and 3d data are supported''')

    lapmat = np.zeros((num_in + num_out,
                       num_in + num_out))
    costmat = get_costmat(pos0, pos1, max_disp)
    m_costmat = ma.masked_invalid(costmat)
    lapmat[:num_in, :num_out] = costmat
    if np.all(np.isnan(costmat)):
        birthcost = deathcost = 1.
    else:
        birthcost = deathcost = np.percentile(m_costmat.compressed(), 90)

    lapmat[num_in:, :num_out] = get_birthmat(pos1, birthcost)
    lapmat[:num_in, num_out:] = get_deathmat(pos0, deathcost)

    fillvalue = m_costmat.max() * 1.05
    lapmat[num_in:, num_out:] = get_lowerright(costmat, fillvalue)

    return lapmat

def get_costmat(pos0, pos1, max_disp):

    distances = cdist(pos0, pos1)
    distances[distances > max_disp] = np.nan
    return distances**2

def get_deathmat(pos0, deathcost):

    deathmat = np.identity(pos0.shape[0]) * deathcost
    deathmat[deathmat == 0] = np.nan
    return deathmat

def get_birthmat(pos1, birthcost):

    birthmat = np.identity(pos1.shape[0]) * birthcost
    birthmat[birthmat == 0] = np.nan
    return birthmat

def get_lowerright(costmat, fillvalue):
    lowerright = costmat.T
    lowerright[np.isfinite(lowerright)] = fillvalue
    return lowerright

def get_test_pos():
    pos0 = np.array([[0, 0, 0],
                 [0, 1, 0],
                 [0, 2, 0]])

    pos1 = pos0.copy()
    pos1[:, 0] += 1
    pos1[:, 1] = [1, 0, 2]
    return pos0, pos1

def get_gap_closing(segments, max_disp0, window_gap=5):

    gc_map = np.zeros((len(segments), len(segments))) * np.nan
    for i, segment0 in enumerate(segments):
        times0 = segment0.index.get_level_values(0)
        last0 = segment0.xs(times0[-1])
        for j, segment1 in enumerate(segments):
            times1 = segment1.index.get_level_values(0)
            delta_t = times1[0] - times0[-1]
            if not (0 < delta_t < window_gap):
                continue
            first1 = segment1.xs(times1[0])
            sqdist01 = ((last0 - first1)**2).sum()
            if sqdist01 > np.sqrt(delta_t * max_disp0):
                continue
            gc_map[i, j] = sqdist01
    return gc_map

def get_splitting(segments):
    pass

def get_merging(segments):
    pass
    
def get_terminating(segments, terminate_cost):
    term_mat = np.identity(len(segments)) * terminate_cost
    term_mat[term_mat == 0] = np.nan
    return term_mat

def get_initiating(segments, init_cost):
    init_mat = np.identity(len(segments)) * init_cost
    init_mat[init_mat == 0] = np.nan
    return init_mat

def get_cmt_mat(segments, max_disp0, window_gap=5):
    
    gc_mat = get_gap_closing(segments, max_disp0, window_gap)
    n_segments = len(segments)
    lapmat = np.zeros((n_segments * 2, n_segments * 2))
    m_costmat = ma.masked_invalid(gc_mat)
    lapmat[:n_segments, :n_segments] = gc_mat
    if np.all(np.isnan(gc_mat)):
        terminate_cost = init_cost = 1.
    else:
        terminate_cost = init_cost = np.percentile(m_costmat.compressed(), 99)
    lapmat[n_segments:, :n_segments] = get_terminating(segments,
                                                       terminate_cost)
    lapmat[:n_segments, n_segments:] = get_initiating(segments,
                                                      init_cost)

    fillvalue = m_costmat.max() * 1.05
    lapmat[n_segments:, n_segments:] = get_lowerright(gc_mat, fillvalue)

    return lapmat

