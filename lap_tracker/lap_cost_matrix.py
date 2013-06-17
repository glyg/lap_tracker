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
        last0 = segment0.iloc[-1]
        for j, segment1 in enumerate(segments):
            times1 = segment1.index.get_level_values(0)
            delta_t = times1[0] - times0[-1]
            if not (0 < delta_t < window_gap):
                continue
            first1 = segment1.iloc[0]
            sqdist01 = ((last0 - first1)**2).sum()
            if sqdist01 > np.sqrt(delta_t * max_disp0):
                continue
            gc_map[i, j] = sqdist01
    return gc_map

def get_splitting(segments, indices, max_disp0, window_gap=5):

    split_lines = []
    for i, segment0 in enumerate(segments):
        times0 = segment0.index.get_level_values(0)
        first_time = times0[0]
        first = segment0.iloc[0]
        for j, segment1 in enumerate(segments):
            times1 = segment1.index.get_level_values(0)
            if not (times1[0] < first_time < times1[-1]):
                continue
            junction = np.where(times1 < first_time)[0][0]
            middle_time = times1[junction]
            delta_t = first_time - middle_time
            if delta_t > window_gap:
                continue
            middle = segment1.loc[middle_time]
            sqdist01 = ((first - middle)**2).sum()
            if sqdist01 > np.sqrt(delta_t * max_disp0):
                continue
            split_lines.append([i, (j, middle_time), sqdist01])
    num_seg = len(segments)
    split_mat = np.zeros((len(indices), num_seg))
    for line in split_lines:
        middle = indices.index((line[1]))
        split_mat[middle, line[0]] = line[2]
    return split_mat


def get_merging(segments, indices, max_disp0, window_gap):

    merge_lines = []
    for i, segment0 in enumerate(segments):
        times0 = segment0.index.get_level_values(0)
        last_time = times0[-1]
        last = segment0.iloc[-1]
        for j, segment1 in enumerate(segments):
            times1 = segment1.index.get_level_values(0)
            if not (times1[0] < last_time < times1[-1]):
                continue
            junction = np.where(times1 < last_time)[0][0]
            middle_time = times1[junction]
            delta_t = last_time - middle_time
            if delta_t > window_gap:
                continue
            middle = segment1.loc[middle_time]
            sqdist01 = ((last - middle)**2).sum()
            if sqdist01 > np.sqrt(delta_t * max_disp0):
                continue
            merge_lines.append([i, (j, middle_time), sqdist01])
    num_seg = len(segments)
    merge_mat = np.zeros((num_seg, len(indices)))
    for line in merge_lines:
        middle = indices.index((line[1]))
        merge_mat[line[0], middle] = line[2]
    return merge_mat

        
def get_terminating(segments, terminate_cost):
    term_mat = np.identity(len(segments)) * terminate_cost
    term_mat[term_mat == 0] = np.nan
    return term_mat

def get_initiating(segments, init_cost):
    init_mat = np.identity(len(segments)) * init_cost
    init_mat[init_mat == 0] = np.nan
    return init_mat

def get_cmt_mat(segments, max_disp0, window_gap=5):
    
    n_segments = len(segments)
    indices = []
    for n, segment in enumerate(segments):
        indices.extend([(n,t) for t in segment.index.get_level_values(0)])
    n_indices = len(indices)
    lapmat = np.zeros((n_segments * 2 + n_indices,
                       n_segments * 2 + n_indices)) * np.nan
    gc_mat = get_gap_closing(segments, max_disp0, window_gap)
    lapmat[:n_segments, :n_segments] = gc_mat
    sm_start = n_segments
    sm_stop = n_segments + n_indices
    lapmat[:n_segments, sm_start:sm_stop] = get_merging(segments,
                                                        indices,
                                                        max_disp0,
                                                        window_gap)
    lapmat[sm_start:sm_stop, :n_segments] = get_splitting(segments,
                                                          indices,
                                                          max_disp0,
                                                          window_gap)
    m_lapmat = ma.masked_invalid(lapmat)
    if np.all(np.isnan(lapmat)):
        terminate_cost = init_cost = 1.
    else:
        terminate_cost = init_cost = np.percentile(m_lapmat.compressed(), 90)
    lapmat[sm_stop:, :n_segments] = get_terminating(segments,
                                                       terminate_cost)
    lapmat[:n_segments, sm_stop:] = get_initiating(segments,
                                                   init_cost)

    fillvalue = m_lapmat.max() * 1.05
    lapmat[sm_stop:, sm_stop:] = get_lowerright(gc_mat, fillvalue)

    return lapmat

