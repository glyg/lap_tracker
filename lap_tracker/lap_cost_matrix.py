# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import logging

import numpy as np

import numpy.ma as ma
import warnings

from .lapjv import lapjv
from .utils.progress import pprogress

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

PERCENTILE = 95


def cost_matrix(pos0, pos1,
                delta_t=1,
                max_disp=1,
                distance_metric="euclidean",
                distance_parameters={},
                cost_function=None):
    """
    Default cost matrix computation :
    1. matrix distance_metric
    2. maximum velocity filter
    3. preprocessing with cost_function
    """

    from scipy.spatial.distance import cdist

    if not cost_function:
        cost_function = lambda x: x

    distances = cdist(pos0, pos1,
                      metric=distance_metric,
                      **distance_parameters)
    distances /= delta_t
    filtered_dist = distances.copy()
    filtered_dist[distances > max_disp] = np.nan

    # self.fillvalue = self.cost_function(p90)
    costmat = cost_function(filtered_dist)
    return costmat


class LAPSolver(object):

    def __init__(self, tracker, verbose=False):
        """
        """

        self.verbose = verbose
        if not self.verbose:
            log.disabled = True
        else:
            log.disabled = False

        self.tracker = tracker
        self.ndims = self.tracker.ndims

        # Cost matrix parameters
        self.cost_function = self.tracker.cost_function
        self.distance_metric = self.tracker.distance_metric
        self.distance_parameters = self.tracker.distance_parameters

        if self.tracker.cost_matrix_function:

            log.info("Custom cost matrix function defined")
            self.get_cost_matrix = self.tracker.cost_matrix_function
            self.cost_matrix_parameters = {}

            if self.tracker.max_cost:
                self.max_cost = self.tracker.max_cost
            else:
                self.max_cost = None
                log.critical("Please define max_cost value or tracking will fail")

        else:

            log.info("Default cost matrix function will be used")
            self.get_cost_matrix = cost_matrix
            self.cost_matrix_parameters = dict(max_disp=self.max_disp,
                                               distance_metric=self.distance_metric,
                                               distance_parameters=self.distance_parameters,
                                               cost_function=self.cost_function)

            if self.cost_function and self.max_disp:
                self.max_cost = self.cost_function(self.max_disp)
            else:
                self.max_cost = None
                log.critical("cost_function and max_disp are needed to compute"
                             " max_cost value")
                log.critical("Tracking will fail")

        self.guessed = True

    @property
    def max_disp(self):
        return self.tracker.max_disp

    def solve(self, *args, **kwargs):

        self.lapmat = self.get_lapmat(*args, **kwargs)

        if np.all(np.isnan(self.lapmat)):
            return None

        idxs_in, idxs_out, costs = self.get_lap_args()
        self.in_links, self.out_links = lapjv(idxs_in, idxs_out, costs)

        return self.in_links, self.out_links

    def get_lap_args(self):

        idxs_in, idxs_out = np.mgrid[:self.lapmat.shape[0],
                                     :self.lapmat.shape[1]]
        idxs_in = idxs_in.flatten()
        idxs_out = idxs_out.flatten()

        flatmat = self.lapmat.flatten()
        finite_flat = np.isfinite(flatmat)
        if not any(finite_flat):
            warnings.warn('No finite element in the LAP matrix')
        costs = flatmat[finite_flat]
        idxs_in = idxs_in[finite_flat]
        idxs_out = idxs_out[finite_flat]
        return idxs_in, idxs_out, costs

    def get_lapmat(self, pos0, pos1, delta_t=1):

        self.pos0 = np.asarray(pos0)
        self.pos1 = np.asarray(pos1)
        if self.pos0.ndim == 1:
            self.pos0 = np.atleast_2d(self.pos0).T
        if self.pos1.ndim == 1:
            self.pos1 = np.atleast_2d(self.pos1).T
        num_in, ndim = self.pos0.shape
        num_out, ndim = self.pos1.shape
        if ndim not in (1, 2, 3):
            raise ValueError('''Only 1d, 2d and 3d data are supported''')

        lapmat = np.zeros((num_in + num_out,
                           num_in + num_out)) * np.nan

        self.costmat = self.get_cost_matrix(self.pos0, self.pos1, delta_t,
                                             **self.cost_matrix_parameters)

        m_costmat = ma.masked_invalid(self.costmat)
        lapmat[:num_in, :num_out] = self.costmat

        ## From TFA (Supplementary note 3):
        ## These alternative costs for “no linking” (b and d in
        ## Fig. 1b) were inferred from the tracking information available
        ## up to the source frame t. They were taken as 1.05 × the
        ## maximal cost of all previous links
        new = m_costmat.max()
        if self.guessed:
            log.info('Getting first value for max cost')
            log.info('Guessed value was: %.4f' % self.max_cost)
            self.max_cost = new
            self.guessed = False
            log.info('New value for max cost: %.4f' % self.max_cost)

        elif np.isfinite(new):
            self.max_cost = max(new, self.max_cost)
            if self.max_cost == new:
                log.info('New value for max cost: %.4f' % (self.max_cost))

        birthcost = deathcost = self.max_cost * 1.05
        lapmat[num_in:, :num_out] = self.get_birthmat(num_out, birthcost)
        lapmat[:num_in, num_out:] = self.get_deathmat(num_in, deathcost)
        m_lapmat = ma.masked_invalid(lapmat)
        self.fillvalue = m_lapmat.max() * 1.05
        lapmat[num_in:, num_out:] = self.get_lowerright()
        return lapmat

    def get_deathmat(self, num_in, deathcost):

        deathmat = np.identity(num_in) * deathcost
        deathmat[deathmat == 0] = np.nan
        return deathmat

    def get_birthmat(self, num_out, birthcost):

        birthmat = np.identity(num_out) * birthcost
        birthmat[birthmat == 0] = np.nan
        return birthmat

    def get_lowerright(self):

        lowerright = self.costmat.T.copy()
        lowerright[np.isfinite(lowerright)] = self.fillvalue
        return lowerright

# def get_test_pos():

#     pos0 = np.array([[0, 0, 0],
#                      [0, 1, 0],
#                      [0, 2, 0]])
#     pos1 = pos0.copy()
#     pos1[:, 0] += 1
#     pos1[:, 1] = [1, 0, 2]
#     return pos0, pos1

    def show_lapmat(self, lapmat=None):
        """
        Show current lap matrix for debugging purpose only
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        if not lapmat:
            m = self.lapmat
        else:
            m = lapmat

        rec_shape = np.array(m.shape)
        size = rec_shape[0]
        num_in, ndim = self.pos0.shape
        num_out, ndim = self.pos1.shape

        # Show matrix
        cax = ax.imshow(m, interpolation='none', cmap='gray',
                        extent=[0, size, 0, size])
        cbar = fig.colorbar(cax)

        # Get nice grid

        # Major axis
        ax.axvline(x=num_out, ymin=0, ymax=size, linewidth=3, color='black')
        ax.axhline(y=size - num_in, xmin=0, xmax=size, linewidth=3,
                   color='black')

        # Minor axis
        for i in range(1, size):
            ax.axvline(x=i, ymin=0, ymax=size, linewidth=1, color='black',
                       alpha=0.7)
            ax.axhline(y=i, xmin=0, xmax=size, linewidth=1, color='black',
                       alpha=0.7)

        # Plot lapjv solutions
        for idx_out, idx_in in enumerate(self.out_links[:self.pos1.shape[0]]):
            ax.scatter(idx_out + 0.5, size - 1 - idx_in + 0.5, marker='o',
                       s=1000, color='green', alpha=0.6)

        ax.set_xlim(0, size)
        ax.set_ylim(0, size)

        # Display nan value
        for p in np.argwhere(np.isnan(m)):
            x = p[1] + 0.5
            y = size - 1 - p[0] + 0.5
            ax.scatter(x, y, marker='x', s=1000, color='red', alpha=0.3)

        ax.grid(False)

        ax.set_xticks(np.arange(0.5, size + 0.5))
        ax.set_yticks(np.arange(0.5, size + 0.5))

        ax.set_xticklabels(np.hstack([np.arange(0, num_out), np.arange(0, num_in)]))
        ax.set_yticklabels(np.hstack([np.arange(0, num_in), np.arange(0, num_out)])[::-1])

        return fig


class CMSSolver(LAPSolver):
    """
    """

    def __init__(self, *args, **kwargs):
        """
        """

        if not kwargs['verbose']:
            log.disabled = True
        else:
            log.disabled = False

        super(CMSSolver, self).__init__(*args, **kwargs)
        self.window_gap = self.tracker.window_gap

        self.segments = [segment[self.tracker.coordinates]
                         for segment in self.tracker.segments()]
        try:
            self.intensities = [segment['I']
                                for segment in self.tracker.segments()]
        except KeyError:
            self.intensities = [(segment['x'] + 1) / (segment['x'] + 1)
                                for segment in self.segments]

        if len(self.segments) == 0:
            warnings.warn('Empty tracker, nothing to do')

    def get_gap_closing(self):

        log.info("Close gap")
        gc_mat = np.zeros((len(self.segments), len(self.segments))) * np.nan
        n = len(self.segments)
        for i, segment0 in enumerate(self.segments):
            if self.verbose:
                pprogress(i / n * 100)
            times0 = segment0.index.get_level_values(0)
            last0 = segment0.iloc[-1]
            for j, segment1 in enumerate(self.segments):
                times1 = segment1.index.get_level_values(0)
                delta_t = times1[0] - times0[-1]
                if not (0 < delta_t <= self.window_gap):
                    continue
                first1 = segment1.iloc[0]
                dist01 = np.sqrt(((last0 - first1)**2).sum())
                if (dist01 / delta_t) > self.max_disp:
                    continue
                gc_mat[i, j] = self.cost_function(dist01)

        if self.verbose:
            pprogress(-1)
        return gc_mat

    def get_splitting(self):

        log.info('Find split')
        split_dic = {}
        n = len(self.segments)
        for i, (segment0, intensities0) in enumerate(zip(self.segments,
                                                         self.intensities)):
            if self.verbose:
                pprogress(i / n * 100)
            times0 = segment0.index
            first_time = times0[0]
            first = segment0.iloc[0]
            for j, (segment1, intensities1) in enumerate(zip(self.segments,
                                                             self.intensities)):
                times1 = segment1.index
                if not (times1[0] < first_time <= times1[-1]):
                    continue
                split = np.where(times1 < first_time)[0][-1]
                split_time = times1[split]
                next_time = times1[split + 1]
                delta_t = first_time - split_time
                if delta_t > self.window_gap:
                    continue
                split = segment1.loc[split_time]
                dist01 = np.sqrt(((first - split)**2).sum())
                if dist01 > delta_t * self.max_disp:
                    continue
                rho01 = (intensities1.loc[split_time]
                         / (intensities1.loc[next_time]
                            + intensities0.loc[first_time]))
                cost01 = self.cost_function(dist01)
                weight = cost01 * rho01 if rho01 > 1. else cost01 * rho01**-2
                split_dic[i, (j, split_time)] =  weight

        if self.verbose:
            pprogress(-1)
        return split_dic

    def get_merging(self):

        log.info('Find merge')
        merge_dic = {}
        n = len(self.segments)
        for i, (segment0, intensities0) in enumerate(
                zip(self.segments, self.intensities)):
            if self.verbose:
                pprogress(i / n * 100)
            times0 = segment0.index
            last_time = times0[-1]
            last = segment0.iloc[-1]
            for j, (segment1, intensities1) in enumerate(
                    zip(self.segments, self.intensities)):
                times1 = segment1.index
                if not (times1[0] <= last_time < times1[-1]):
                    continue
                junction = np.where(times1 > last_time)[0][0]
                merge_time = times1[junction]
                prev_time = times1[junction -1]
                delta_t = merge_time - last_time
                if delta_t > self.window_gap:
                    continue
                merge = segment1.loc[merge_time]
                dist01 = np.sqrt(((last - merge)**2).sum())
                if dist01 > (delta_t * self.max_disp):
                    continue
                rho01 = (intensities1.loc[merge_time]
                         / (intensities1.loc[prev_time]
                            + intensities0.loc[last_time]))
                cost01 = self.cost_function(dist01)
                weight = cost01 * rho01 if rho01 > 1. else cost01 * rho01**-2
                merge_dic[i, (j, merge_time)] =  weight

        if self.verbose:
            pprogress(-1)
        return merge_dic

    def get_cms_seeds(self):

        seeds = [key[1] for key in self.split_dic.keys()]
        seeds.extend(key[1] for key in self.merge_dic.keys())
        seeds = np.unique(seeds)
        return [tuple(seed) for seed in seeds]

    def get_alt_merge_split(self):

        alt_merge_mat = np.zeros((len(self.seeds),
                                  len(self.seeds))) * np.nan
        alt_split_mat = alt_merge_mat.copy()
        avg_disps = np.array([np.sqrt((segment.diff().dropna()**2
                                   ).sum(axis=1)).mean(axis=0)
                              for segment in self.segments])
        global_mean = avg_disps[np.isfinite(avg_disps)].mean()
        avg_disps[np.isnan(avg_disps)] = global_mean
        avg_disps = self.cost_function(avg_disps)

        for n, seed in enumerate(self.seeds):
            seg_index = seed[0]
            pos_index = seed[1]
            #intensity previous to merge/split
            intensity = self.intensities[seg_index].loc[:pos_index]
            if intensity.shape[0] == 1:
                merge_factor = split_factor = 1.
            else:
                i0, i1 = intensity.iloc[-2:]
                if i1 / i0 > 1:
                    merge_factor = i1 / i0
                    split_factor = (i0 / i1)**-2
                else:
                    merge_factor = (i1 / i0)**-2
                    split_factor = i0 / i1

            alt_merge_mat[n, n] = (avg_disps[seg_index]
                                   * merge_factor)
            alt_split_mat[n, n] = (avg_disps[seg_index]
                                   * split_factor)
        return alt_merge_mat, alt_split_mat

    def get_lapmat(self, gap_close_only=False,
                   verbose=False):

        if not verbose:
            log.disabled = True
        else:
            log.disabled = False

        n_segments = len(self.segments)

        self.gc_mat = self.get_gap_closing()
        self.split_dic = self.get_splitting()
        self.merge_dic = self.get_merging()
        self.seeds = self.get_cms_seeds()
        n_seeds = len(self.seeds)
        if not gap_close_only:
            self.split_mat = np.zeros((len(self.seeds),
                                       len(self.segments))) * np.nan
            self.merge_mat = self.split_mat.copy().T
            for key, weight in self.split_dic.items():
                i = key[0]
                j = self.seeds.index(key[1])
                self.split_mat[j, i] = weight
            for key, weight in self.merge_dic.items():
                i = key[0]
                j = self.seeds.index(key[1])
                self.merge_mat[i, j] = weight

        sm_start = n_segments
        sm_stop = n_segments + n_seeds

        ### Looking at figure 1c from TFA one woulfd think that
        ### the matrix shape is (n_segments + n_seeds + n_segments, n_segments * 2 + n_seeds)
        ### Actually, the upper right and lower left blocks have shape (n_segments+ n_seeds)
        ### to give space for alternate costs d' and b'. So overall shape is
        ### ((n_segments + n_seeds)*2, (n_segments + n_seeds)*2) ....
        size = (n_segments + n_seeds) * 2
        lapmat = np.zeros((size, size)) * np.nan
        lapmat[:n_segments, :n_segments] = self.gc_mat
        alt_sm_start = size  - n_seeds

        if not gap_close_only:
            lapmat[:n_segments, sm_start:sm_stop] = self.merge_mat

            lapmat[sm_start:sm_stop, :n_segments] = self.split_mat

            alt_merge_mat, alt_split_mat = self.get_alt_merge_split()

            lapmat[sm_start:sm_stop, alt_sm_start:] = alt_split_mat
            lapmat[alt_sm_start:, sm_start:sm_stop] = alt_merge_mat

        m_lapmat = ma.masked_invalid(lapmat)
        if np.all(np.isnan(lapmat)):
            terminate_cost = init_cost = 1.
            warnings.warn('all costs are invalid')
        else:
            terminate_cost = init_cost = np.percentile(m_lapmat.compressed(),
                                                       PERCENTILE) * 4.

        lapmat[:n_segments, sm_stop:alt_sm_start] = self.get_birthmat(n_segments,
                                                                  init_cost)
        lapmat[sm_stop:alt_sm_start, :n_segments] = self.get_deathmat(n_segments,
                                                                  terminate_cost)
        m_lapmat = ma.masked_invalid(lapmat)
        self.fillvalue = m_lapmat.max() * 1.05
        self.costmat = lapmat[:sm_stop, :sm_stop]
        lapmat[sm_stop:, sm_stop:] = self.get_lowerright()
        if gap_close_only:
            red_lapmat = np.zeros((n_segments * 2, n_segments *2))
            red_lapmat[:n_segments, :n_segments] = lapmat[:n_segments,
                                                          :n_segments]
            red_lapmat[n_segments:, :n_segments] = lapmat[sm_stop:alt_sm_start,
                                                          :n_segments]
            red_lapmat[:n_segments, n_segments:] = lapmat[:n_segments,
                                                          sm_stop:alt_sm_start]
            red_lapmat[n_segments:, n_segments:] = lapmat[sm_stop:alt_sm_start,
                                                          sm_stop:alt_sm_start]
            return red_lapmat
        return lapmat
