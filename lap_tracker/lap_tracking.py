#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.gaussian_process import GaussianProcess
import warnings

from .lap_cost_matrix import get_lapmat, get_lap_args, get_cmt_mat
from .lapjv import lapjv

class LAPTracker():

    def __init__(self, track_df, hdfstore,
                 x_scale=1., y_scale=1., z_scale=1.):
        self.track = track_df
        self.store = hdfstore
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.z_scale = z_scale
        self.track['x'] *= self.x_scale
        self.track['y'] *= self.y_scale
        self.track['z'] *= self.z_scale

        
    @property
    def times(self):
        '''Unique values of the level 0 index of `self.track`'''
        return self.track.index.get_level_values(0).unique()

    @property
    def labels(self):
        '''Unique values of the level 1 index of `self.track`'''
        return self.track.index.get_level_values(1).unique()
        
    def get_track(self, max_disp, window_gap, ndims=3):
        
        self.track['new_label'] = self.track.index.get_level_values(1)
        time_points = self.times
        for t0, t1 in zip(time_points[:-1], time_points[1:]):
            self.position_track(t0, t1, max_disp, ndims)
        self.track.set_index('new_label', append=True, inplace=True)
        self.track.reset_index(level='label', drop=True, inplace=True)
        self.track.index.names[1] = 'label'
        self.close_merge_split(max_disp, window_gap)
        self.store.open()
        self.store['sorted'] = self.track
        self.store.close()

    def close_merge_split(self, max_disp, window_gap):
        segments = [segment[['x', 'y']]
                    for segment in self.segments()]
        lapmat = get_cmt_mat(segments, max_disp, window_gap)
        idxs_in, idxs_out, costs = get_lap_args(lapmat)
        in_links, out_links = lapjv(idxs_in, idxs_out, costs)
        num_seqs = len(segments)
        old_labels = self.track.index.get_level_values(1).values
        new_labels = old_labels.copy()
        unique_old = np.unique(old_labels)
        unique_new = np.unique(new_labels)
        for n, idx_in in enumerate(out_links[:num_seqs]):
            if idx_in >= num_seqs:
                # new segment
                new_label = unique_new.max() + 1
            else:
                new_label  = unique_new[idx_in]
            unique_new[n] = new_label
        for old, new in zip(unique_old, unique_new):
            new_labels[old_labels == old] = new
        self.track['new_label'] = new_labels
        self.track.set_index('new_label', append=True, inplace=True)
        self.track.reset_index(level='label', drop=True, inplace=True)
        self.track.index.names[1] = 'label'

    def position_track(self, t0, t1, max_disp, ndims=3):
        if ndims == 3:
            pos1 = self.track.xs(t1)[['x', 'y', 'z']]
        elif ndims == 2:
            pos1 = self.track.xs(t1)[['x', 'y']]
        pos0, mse0 = self.predict_positions(t0, t1, ndims)
        # max_disp = 
        lapmat = get_lapmat(pos0, pos1, max_disp * (t1 - t0))
        idxs_in, idxs_out, costs = get_lap_args(lapmat)
        try:
            in_links, out_links = lapjv(idxs_in, idxs_out, costs)
        except AssertionError:
            warnings.warn('''Someting's' amiss between points %s and %s'''
                          % (t0, t1), RuntimeWarning)
            for n in range(pos1.shape[0]):
                new_label = self.track['new_label'].max() + 1
                self.track.xs(t1)['new_label'][n] = new_label
            return
        for n, idx_in in enumerate(out_links[:pos1.shape[0]]):
            if idx_in >= pos0.shape[0]:
                # new segment
                new_label = self.track['new_label'].max() + 1
            else:
                new_label  = self.track.xs(t0)['new_label'][idx_in]
            self.track.xs(t1)['new_label'][n] = new_label
    
    def predict_positions(self, t0, t1, ndims=3):
        """
        """
        ## Possible values for corr
        # 'absolute_exponential', 'squared_exponential',
        # 'generalized_exponential', 'cubic', 'linear'
        corr = 'absolute_exponential'
        regr = 'quadratic'
        theta0 = 0.1
        gp_kwargs = {'corr':corr,
                     'regr':regr,
                     'theta0':theta0}

        coordinates = ['x', 'y'] if ndims==2 else ['x', 'y', 'z']
        pos0 = self.track.xs(t0)[coordinates]
        mse0 = pos0.copy()
        
        if np.where(self.times == t1) < 3:
            return pos0, mse0 * 0.
        for lbl in self.labels:
            
            segment = self.get_segment(lbl).loc[:t0]
            if segment.shape[0] == 0:
                continue
            if not t0 in segment.index:
                continue
            times = np.atleast_2d(segment.index.get_level_values(0))
            if len(times) < 3:
                pos = segment[coordinates].loc[t0]
                mse = pos * 0
            else:
                pred = [self._predict_coordinate(segment, coord, times,
                                                 t1, **gp_kwargs)
                        for coord in coordinates]
                pos = [p[0] for p in pred]
                mse = [p[1] for p in pred]
            pos0.ix[lbl] = pos
            mse0.ix[lbl] = mse
        return pos0, mse0

    def _predict_coordinate(self, segment, coord, times, t1, **kwargs):

        prev = segment[coord]
        nugget = prev.std()**2 
        gp = GaussianProcess(nugget=nugget, **kwargs)
        gp.fit(times, prev)
        return gp.predict(t1, eval_MSE=True)

    def remove_shorts(self, min_length=3):
        labels = self.track.index.get_level_values(1).unique()
        for lbl in labels:
            segment = self.get_segment(lbl)
            if segment.shape[0] < min_length:
                self.track = self.track.drop(lbl, level=1)

    def get_segment(self, lbl):
        return self.track.xs(lbl, level=1)

    def segments(self):
        labels = self.track.index.get_level_values(1).unique()        
        for lbl in labels:
            yield self.get_segment(lbl)

    def show_3D(self):

        fig, axes = plt.subplots(1, 2, subplot_kw={'projection':'3d'})
        ax0, ax1 = axes
        for label in self.labels:
            ax0, ax1 = self.show_segment(label, axes)
        return ax0, ax1

    def show_segment(self, label, axes=None):
        if axes is None:
            fig, axes = plt.subplots(1, 2, subplot_kw={'projection':'3d'})
        ax0, ax1 = axes
        segment = self.get_segment(label)
        times = segment.index.get_level_values(0)
        ax0.plot(times, segment['x'],
                 zs=segment['y'])
        colors = plt.cm.jet(segment['x'].size)
        ax1.plot(segment['x'], segment['y'],
                 zs=segment['z'])
        ax1.scatter(segment['x'], segment['y'],
                    segment['z'], c=colors)
        ax0.set_xlabel('Time (min)')
        ax0.set_ylabel(u'x position (µm)')
        ax0.set_zlabel(u'y position (µm)')
        ax1.set_xlabel(u'x position (µm)')
        ax1.set_ylabel(u'y position (µm)')
        ax1.set_zlabel(u'z position (µm)')
        return ax0, ax1

    