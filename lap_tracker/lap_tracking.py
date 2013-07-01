#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.gaussian_process import GaussianProcess
from sklearn.decomposition import PCA
import warnings

from .lap_cost_matrix import get_lapmat, get_lap_args, get_cmt_mat
from .lapjv import lapjv

DEFAULTS = {'max_disp':0.1,
            'window_gap':10,
            'sigma':1.,
            'ndims':3,
            'gp_corr':'squared_exponential',
            'gp_regr':'quadratic',
            'gp_theta0':0.1}

class LAPTracker(object):

    def __init__(self, track_df, hdfstore, params=DEFAULTS):
        self.track = track_df
        self.store = hdfstore
        self.params = params
        # Complete the parameter by the defaults
        for key, value in DEFAULTS.items():
            if not self.params.has_key(key):
                self.params[key] = value
        self.gp_kwargs = {}
        for key, value in params.items():
            if isinstance(key, str) :
                if key.startswith('gp_'):
                    self.gp_kwargs[key[3:]] = value
                else:
                    self.__setattr__(key, value)
        self.dist_function = np.square
        
                    
    @property
    def times(self):
        '''Unique values of the level 0 index of `self.track`'''
        return self.track.index.get_level_values(0).unique()

    @property
    def labels(self):
        '''Unique values of the level 1 index of `self.track`'''
        return self.track.index.get_level_values(1).unique()

    def get_track(self, **kwargs):

        for key, value in kwargs.items():
            if key.startswith('gp_'):
                self.gp_kwargs[key[3:]] = value
            else:
                self.__setattr__(key, value)

        self.track['new_label'] = self.track.index.get_level_values(1)
        time_points = self.times
        for t0, t1 in zip(time_points[:-1], time_points[1:]):
            self.position_track(t0, t1)
        self.track.set_index('new_label', append=True, inplace=True)
        self.track.reset_index(level='label', drop=True, inplace=True)
        self.track.index.names[1] = 'label'
        self.store.open()
        self.store['sorted'] = self.track
        self.store.close()

    def reverse_track(self):
        
        self.track['rev_times'] = self.track.index.get_level_values(0)
        self.track['rev_times'] = (self.track['rev_times'].iloc[-1]
                                   - self.track['rev_times'])
        self.track = self.track.iloc[::-1]        
        self.track.set_index('rev_times', append=True, inplace=True,
                             drop='True')
        self.track.reset_index(level='time_stamp', drop=True, inplace=True)
        self.track = self.track.swaplevel(0, 1, axis=0)
        self.track.index.names[0] = 'time_stamp'

        
    def close_merge_split(self):
        
        if self.ndims == 2:
            segments = [segment[['x', 'y']]
                        for segment in self.segments()]
        elif self.ndims == 3:
            segments = [segment[['x', 'y', 'z']]
                        for segment in self.segments()]
        try:
            intensities = [segment['I'] for segment in self.segments()]
        except KeyError:
            intensities = [(segment['x'] + 1) / (segment['x'] + 1)
                           for segment in self.segments()]
        lapmat = get_cmt_mat(segments, intensities,
                             self.max_disp, self.window_gap,
                             gap_close_only=True)
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
        return lapmat

        
    def position_track(self, t0, t1):

        coordinates = ['x', 'y'] if self.ndims == 2 else ['x', 'y', 'z']

        pos1 = self.track.xs(t1)[coordinates]
        if self.predict:
            pos0, mse0 = self.predict_positions(t0, t1)
        else:
            pos0 = self.track.xs(t0)[coordinates]
        lapmat = get_lapmat(pos0, pos1,
                            self.max_disp * (t1 - t0),
                            self.dist_function)
        idxs_in, idxs_out, costs = get_lap_args(lapmat)
        try:
            in_links, out_links = lapjv(idxs_in, idxs_out, costs)
        except AssertionError:
            warnings.warn('''Someting's' amiss between points %s and %s'''
                          % (t0, t1), RuntimeWarning)
            for n in range(pos1.shape[0]):
                new_label = self.track['new_label'].max() + 1
                self.track.xs(t1)['new_label'].iloc[n] = new_label
            return
        for n, idx_in in enumerate(out_links[:pos1.shape[0]]):
            if idx_in >= pos0.shape[0]:
                # new segment
                new_label = self.track['new_label'].max() + 1
            else:
                new_label  = self.track.xs(t0)['new_label'].iloc[idx_in]
            self.track.xs(t1)['new_label'].iloc[n] = new_label
    
    def predict_positions(self, t0, t1):
        """
        """
        
        coordinates = ['x', 'y'] if self.ndims == 2 else ['x', 'y', 'z']
        pos0 = self.track.xs(t0)[coordinates]
        mse0 = pos0.copy()
        
        if np.where(self.times == t1) < 3:
            return pos0, mse0 * 0.
        for lbl in self.labels:
            try:
                segment = self.get_segment(lbl).iloc[t0]
            except KeyError:
                continue
            if segment.shape[0] == 0:
                continue
            if not t0 in segment.index:
                continue
            times = segment.index.get_level_values(0)
            if times.size < 3:
                pos = segment[coordinates].loc[t0]
                mse = pos * 0
            else:
                pred = [_predict_coordinate(segment, coord, times,
                                            t0, self.sigma,
                                            **self.gp_kwargs)
                        for coord in coordinates]
                pos = [p[0] for p in pred]
                mse = [p[1] for p in pred]
            pos0.ix[lbl] = pos
            mse0.ix[lbl] = mse
        return pos0, mse0

    def remove_shorts(self, min_length=3):
        labels = self.track.index.get_level_values(1).unique()
        for lbl in labels:
            segment = self.get_segment(lbl)
            if segment.shape[0] < min_length:
<<<<<<< HEAD
                self.track = self.track.drop([lbl,], level=1)
=======
                self.track = self.track.drop(lbl, level=1)
>>>>>>> 0b8b85200b099ca6101ad049cf6e97c3435e1c60

    def get_segment(self, lbl):
        return self.track.xs(lbl, level=1)

    def segments(self):
<<<<<<< HEAD
        for lbl in self.labels:
=======
        labels = self.track.index.get_level_values(1).unique()        
        for lbl in labels:
>>>>>>> 0b8b85200b099ca6101ad049cf6e97c3435e1c60
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

<<<<<<< HEAD
    def do_pca(self, ndims=3, centered=True):
        
        self.pca = PCA()
=======
    def do_pca(self, ndims=3):
        
        pca = PCA()
>>>>>>> 0b8b85200b099ca6101ad049cf6e97c3435e1c60
        if ndims == 2:
            coords = ['x', 'y']
            pca_coords = ['x_pca', 'y_pca']
        elif ndims == 3:
            coords = ['x', 'y', 'z']
            pca_coords = ['x_pca', 'y_pca', 'z_pca']
<<<<<<< HEAD
        rotated = self.pca.fit_transform(self.track[coords])
        for n, coord in enumerate(pca_coords):
            self.track[coord] = rotated[:, n]
        if centered:
            center = self.track[pca_coords].mean(level=0)
            center = center.reindex(self.track.index, level=0)
            for coord in pca_coords:
                self.track[coord] -= center[coord]
=======
        rotated = pca.fit_transform(self.track[coords])
        for n, coord in enumerate(pca_coords):
            self.track[coord] = rotated[:, n]
        center = self.track[pca_coords].mean(level=0)
        center = center.reindex(self.track.index, level=0)
        for coord in pca_coords:
            self.track[coord] -= center[coord]
>>>>>>> 0b8b85200b099ca6101ad049cf6e97c3435e1c60

        
def _predict_coordinate(segment, coord, times, t1, sigma=10., **kwargs):

    times = np.atleast_2d(times).T
    prev = segment[coord]
    nugget = (sigma / (prev + sigma))**2 
    gp = GaussianProcess(nugget=nugget, **kwargs)
    gp.fit(times, prev)
    return gp.predict(t1, eval_MSE=True)
