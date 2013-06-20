#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

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
    

def get_lapmat(pos0, pos1, max_disp=1000., dist_function=np.square):

    pos0 = np.asarray(pos0)
    pos1 = np.asarray(pos1)
    num_in, ndim = pos0.shape
    num_out, ndim = pos1.shape
    if ndim not in (2, 3):
        raise ValueError('''Only 2d and 3d data are supported''')

    lapmat = np.zeros((num_in + num_out,
                       num_in + num_out))
    costmat = get_costmat(pos0, pos1, max_disp, dist_function)
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

def get_costmat(pos0, pos1, max_disp, dist_function=np.square):

    distances = cdist(pos0, pos1)
    distances[distances > max_disp] = np.nan
    return dist_function(distances)

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

def get_splitting(segments, intensities, max_disp0, window_gap=5):

    split_dic = {}
    for i, (segment0, intensities0) in enumerate(zip(segments, intensities)):
        times0 = segment0.index.get_level_values(0)
        first_time = times0[0]
        first = segment0.iloc[0]
        for j, (segment1, intensities1) in enumerate(zip(segments,
                                                         intensities)):
            times1 = segment1.index.get_level_values(0)
            if not (times1[0] < first_time < times1[-1]):
                continue
            junction = np.where(times1 < first_time)[0][0]
            middle_time = times1[junction]
            next_time = times1[junction + 1]
            delta_t = first_time - middle_time
            if delta_t > window_gap:
                continue
            middle = segment1.loc[middle_time]
            sqdist01 = ((first - middle)**2).sum()
            if sqdist01 > np.sqrt(delta_t * max_disp0):
                continue
            rho01 = (intensities1.loc[middle_time]
                     / (intensities1.loc[next_time]
                        + intensities0.loc[first_time]))
            weight = sqdist01 * rho01 if rho01 > 1. else sqdist01 * rho01**-2
            split_dic[i, (j, middle_time)] =  weight
    return split_dic

def get_merging(segments, intensities, max_disp0, window_gap):

    merge_dic = {}
    for i, (segment0, intensities0) in enumerate(zip(segments, intensities)):
        times0 = segment0.index.get_level_values(0)
        last_time = times0[-1]
        last = segment0.iloc[-1]
        for j, (segment1, intensities1) in enumerate(zip(segments,
                                                         intensities)):
            times1 = segment1.index.get_level_values(0)
            if not (times1[0] < last_time < times1[-1]):
                continue
            junction = np.where(times1 > last_time)[0][0]
            middle_time = times1[junction]
            prev_time = times1[junction -1]
            delta_t = last_time - middle_time
            if delta_t > window_gap:
                continue
            middle = segment1.loc[middle_time]
            sqdist01 = ((last - middle)**2).sum()
            if sqdist01 > np.sqrt(delta_t * max_disp0):
                continue
            rho01 = (intensities1.loc[middle_time]
                     / (intensities1.loc[prev_time]
                        + intensities0.loc[last_time]))
            weight = sqdist01 * rho01 if rho01 > 1. else sqdist01 * rho01**-2
            merge_dic[i, (j, middle_time)] =  weight
    return merge_dic

def get_alt_merge_split(segments, seeds, intensities,
                        split_dic, merge_dic):
    
    alt_merge_mat = np.zeros((len(segments), len(seeds))) * np.nan
    alt_split_mat = alt_merge_mat.T
    avg_disps = [np.sqrt((segment.diff().dropna()*2).sum(axis=1))
                 for segment in segments]
    avg_disps = np.array([
        np.sqrt((segment.diff().dropna()*2).sum(axis=1).mean(axis=0))
        for segment in segments])
    global_mean = avg_disps[np.isfinite(avg_disps)].mean()
    avg_disps[np.isnan(avg_disps)] = global_mean
    for n, seed in enumerate(seeds):
        seg_index = seed[0]
        pos_index = seed[1]
        #intensity previous to merge/split
        intensity = intensities[seg_index].loc[:pos_index]
        if intensity.shape[0] == 1:
            if merge_dic.has_key(seed):
                alt_merge_mat[seg_index, n] = avg_disps[seg_index]
            if split_dic.has_key(seed):
                alt_split_mat[n, seg_index] = avg_disps[seg_index]
        else:
            i0, i1 = intensity.iloc[-2:]
            if i1 / i0 > 1 :
                alt_merge_mat[seg_index, n] = (avg_disps[seg_index]
                                               * (i1 / i0))
                alt_split_mat[n, seg_index] = (avg_disps[seg_index]
                                               * (i0 / i1)**-2)
            else:
                alt_merge_mat[seg_index, n] = (avg_disps[seg_index]
                                               * (i1 / i0)**-2)
                alt_split_mat[n, seg_index] = (avg_disps[seg_index]
                                               * (i0 / i1))
    return alt_merge_mat, alt_split_mat

            
def get_terminating(segments, terminate_cost):
    term_mat = np.identity(len(segments)) * terminate_cost
    term_mat[term_mat == 0] = np.nan
    return term_mat

def get_initiating(segments, init_cost):
    init_mat = np.identity(len(segments)) * init_cost
    init_mat[init_mat == 0] = np.nan
    return init_mat

def get_cmt_mat(segments, intensities, max_disp0, window_gap=5):
    
    n_segments = len(segments)
    gc_mat = get_gap_closing(segments, max_disp0, window_gap)
    split_dic = get_splitting(segments, intensities, 0.4, 5)
    merge_dic = get_merging(segments, intensities, 0.4, 5)
    seeds = [key[1] for key in split_dic.keys()]
    seeds.extend(key[1] for key in merge_dic.keys())
    seeds = np.unique(seeds)
    
    seeds = [tuple(seed) for seed in seeds]
    split_mat = np.zeros((len(seeds), len(segments))) * np.nan
    merge_mat = split_mat.copy().T
    for key, weight in split_dic.items():
        i = key[0]
        j = seeds.index(key[1])
        split_mat[j, i] = weight
    for key, weight in merge_dic.items():
        i = key[0]
        j = seeds.index(key[1])
        merge_mat[i, j] = weight
    sm_start = n_segments
    n_seeds = len(seeds)
    sm_stop = n_segments + n_seeds
    lapmat = np.zeros((n_segments * 2 + n_seeds,
                       n_segments * 2 + n_seeds)) * np.nan
    lapmat[:n_segments, :n_segments] = gc_mat
    lapmat[:n_segments, sm_start:sm_stop] = merge_mat
    lapmat[sm_start:sm_stop, :n_segments] = split_mat
    alt_merge_mat, alt_split_mat = get_alt_merge_split(segments,
                                                       seeds,
                                                       intensities,
                                                       split_dic,
                                                       merge_dic)
    lapmat[sm_start:sm_stop, sm_stop:] = alt_split_mat
    lapmat[sm_stop:, sm_start:sm_stop] = alt_merge_mat
    
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

