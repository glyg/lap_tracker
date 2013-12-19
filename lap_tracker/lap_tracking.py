# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import logging
import traceback

import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcess
from sklearn.decomposition import PCA
import warnings

from .lap_cost_matrix import LAPSolver, CMSSolver
from .lapjv import lapjv
from .utils.progress import pprogress

log = logging.getLogger(__name__)

DEFAULTS = {'max_disp': 0.1,
            'window_gap': 10,
            'sigma': 1.,
            'ndims': 3,
            'gp_corr': 'squared_exponential',
            'gp_regr': 'quadratic',
            'gp_theta0': 0.1,
            'distance_metric': 'euclidean',
            'distance_parameters': {}}


class LAPTracker(object):

    def __init__(self, track_df=None,
                 hdfstore=None,
                 coords=['x', 'y', 'z'],
                 cost_function=np.square,
                 params=DEFAULTS,
                 verbose=True):

        self.verbose = verbose
        if not self.verbose:
            log.disabled = True
        else:
            log.disabled = False

        self.coordinates = coords
        self.track = track_df
        try:
            self.track.index.set_names(['t', 'label'], inplace=True)
        except AttributeError:
            self.track.index.names = ['t', 'label']
        self.store = hdfstore
        self.load_parameters(params)
        self.cost_function = cost_function
        self.pos_solver = LAPSolver(self, verbose=self.verbose)

    def load_parameters(self, params):
        """
        """
        self.params = params

        # Complete the parameter by the defaults
        for key, value in DEFAULTS.items():
            if key not in self.params.keys():
                self.params[key] = value
        self.gp_kwargs = {}
        for key, value in self.params.items():
            if isinstance(key, str) or isinstance(key, unicode):
                if key.startswith('gp_'):
                    self.gp_kwargs[key[3:]] = value
                else:
                    self.__setattr__(key, value)

    @property
    def times(self):
        '''Unique values of the level 0 index of `self.track`'''
        return self.track.index.get_level_values(0).unique()

    @property
    def labels(self):
        '''Unique values of the level 1 index of `self.track`'''
        return self.track.index.get_level_values(1).unique()

    def get_track(self, verbose=False, save=True, **kwargs):

        for key, value in kwargs.items():
            if key.startswith('gp_'):
                self.gp_kwargs[key[3:]] = value
            else:
                self.__setattr__(key, value)

        log.info('Get track (predict=%s)' % str(self.predict))
        old_label = self.track.index.get_level_values(1).values
        self.track['new_label'] = old_label.astype(np.float)
        time_points = self.times

        n = len(time_points) - 1
        for i, (t0, t1) in enumerate(zip(time_points[:-1], time_points[1:])):
            if verbose:
                pprogress(i / n * 100)
            self.position_track(t0, t1)

        if verbose:
            pprogress(-1)

        self.track.set_index('new_label', append=True, inplace=True)
        self.track.reset_index(level='label', drop=True, inplace=True)
        try:
            self.track.index.set_names(['t', 'label'], inplace=True)
        except AttributeError:
            self.track.index.names = ['t', 'label']
        if save:
            self.save_df(self.track, 'sorted')
        self.track.sortlevel('label', inplace=True)
        self.track.sortlevel('t', inplace=True)
        relabel_fromzero(self.track, 'label', inplace=True)

    def save_df(self, dataframe, name):
        try:
            self.store.open()
            self.store[name] = dataframe
            self.store.close()
        except AttributeError:
            warnings.warn('''No store has been provided, can't save''')

    def reverse_track(self):

        self.track['rev_times'] = self.track.index.get_level_values(0)
        self.track['rev_times'] = (self.track['rev_times'].iloc[-1]
                                   - self.track['rev_times'])
        self.track = self.track.iloc[::-1]
        self.track.set_index('rev_times', append=True,
                             inplace=True, drop='True')
        self.track.reset_index(level='t', drop=True, inplace=True)
        self.track = self.track.swaplevel(0, 1, axis=0)
        try:
            self.track.index.set_names(['t', 'label'], inplace=True)
        except AttributeError:
            self.track.index.names = ['t', 'label']
        self.track.sortlevel('label', inplace=True)
        self.track.sortlevel('t', inplace=True)

    def close_merge_split(self, gap_close_only=True, save=True):

        try:
            self.cms_solver = CMSSolver(self, verbose=self.verbose)
        except TypeError as e:
            log.critical("Python darkness get you")
            log.critical("Unable to perform close/merge/split")
            log.critical("You should restart your kernel/interpreter")
            log.critical("\n" + traceback.format_exc())
            return None

        in_links, out_links = self.cms_solver.solve(
            gap_close_only=gap_close_only)
        n_segments = len(self.cms_solver.segments)
        n_seeds = len(self.cms_solver.seeds)
        sm_start = n_segments
        sm_stop = n_segments + n_seeds if not gap_close_only else n_segments

        ## First split and merge, because this changes
        ## data length, without changing the unique labels

        labels = self.labels
        for n, idx_in in enumerate(out_links[:n_segments]):
            ## splitting
            if n_segments <= idx_in < sm_stop:
                seed = self.cms_solver.seeds[idx_in - sm_start]
                root_label = labels[seed[0]]
                split_time = seed[1]
                branch_label = labels[n]
                self.split(root_label, split_time, branch_label)

        for n, idx_in in enumerate(out_links[sm_start:sm_stop]):
            ## merging
            if idx_in < n_segments:
                seed = self.cms_solver.seeds[n]
                root_label = labels[seed[0]]
                merge_time = seed[1]
                branch_label = labels[idx_in]
                self.merge(root_label, merge_time, branch_label)

        ## Now for gap closing
        old_labels = self.track.index.get_level_values(1).values
        new_labels = old_labels.copy()
        unique_old = np.unique(old_labels)
        unique_new = np.unique(new_labels)
        for n, idx_in in enumerate(out_links[:n_segments]):
            ## gap closing
            if idx_in < n_segments:
                new_label = unique_new[idx_in]
                unique_new[n] = new_label
                log.info('Gap closing for segment %i'
                         % new_label)
            elif idx_in >= sm_stop:
                unique_new[n] = unique_new.max() + 1
        for old, new in zip(unique_old, unique_new):
            new_labels[old_labels == old] = new

        self.track['new_label'] = new_labels
        self.track.set_index('new_label', append=True, inplace=True)
        self.track.reset_index(level='label', drop=True, inplace=True)
        try:
            self.track.index.set_names(['t', 'label'], inplace=True)
        except AttributeError:
            self.track.index.names = ['t', 'label']
        self.track.sortlevel('label', inplace=True)
        self.track.sortlevel('t', inplace=True)
        relabel_fromzero(self.track, 'label', inplace=True)
        if save:
            self.save_df(self.track, 'sorted')

    def split(self, root_label, split_time, branch_label):

        log.info('''Splitting segment %i @ time %i '''
                 % (int(root_label), split_time))
        root_segment = self.get_segment(root_label)
        try:
            root_segment['I'] / + 2.
        except KeyError:
            pass
        duplicated = root_segment.loc[:split_time].copy()
        dup_index = pd.MultiIndex.from_tuples([(t, branch_label)
                                               for t in duplicated.index])
        duplicated.set_index(dup_index, inplace=True)
        self.track = self.track.append(duplicated)
        self.track.sortlevel(0, inplace=True)

    def relabel_fromzero(self):
        relabel_fromzero(self.track, 'label', inplace=True)

    def merge(self, root_label, merge_time, branch_label):

        log.info('''Merge root %i @ time %i ''' % (int(root_label), merge_time))
        root_segment = self.get_segment(root_label)
        duplicated = root_segment.loc[merge_time:].copy()
        dup_index = pd.MultiIndex.from_tuples([(t, branch_label)
                                               for t in duplicated.index])
        duplicated.set_index(dup_index, inplace=True)
        self.track = self.track.append(duplicated)
        self.track.sortlevel(0, inplace=True)

    def position_track(self, t0, t1):

        pos1 = self.track.loc[t1][self.coordinates]

        if self.predict:
            pos0, mse0 = self.predict_positions(t0, t1)
        else:
            pos0 = self.track.loc[t0][self.coordinates]
        delta_t = t1 - t0
        results = self.pos_solver.solve(pos0, pos1, delta_t)

        if not results:
            log.warning("LAP matrice is invalid for t = "
                        "%d and t = %d (check max_disp value)" % (t0, t1))
            return None

        in_links, out_links = results

        for idx_out, idx_in in enumerate(out_links[:pos1.shape[0]]):
            if idx_in >= pos0.shape[0]:
                # new segment
                new_label = self.track['new_label'].max() + 1.
            else:
                new_label = self.track.loc[t0]['new_label'].iloc[idx_in]
            self.track.loc[t1, 'new_label'].iloc[idx_out] = new_label

    def predict_positions(self, t0, t1):
        """
        """

        pos0 = self.track.xs(t0)[self.coordinates]
        mse0 = pos0.copy() * 0.

        if np.where(self.times == t1) < 3:
            return pos0, mse0
        for lbl in self.labels:
            try:
                segment = self.get_segment(lbl).loc[:t0]
            except KeyError:
                continue
            if segment.shape[0] == 0:
                continue
            if not t0 in segment.index:
                continue
            times = segment.index.get_level_values(0)
            if times.size < 3:
                pos = segment[self.coordinates].loc[t0]
                mse = pos * 0
            else:
                pred = [_predict_coordinate(segment, coord, times,
                                            t1, self.sigma,
                                            **self.gp_kwargs)
                        for coord in self.coordinates]
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
                self.track = self.track.drop([lbl, ], level=1)

    def get_segment(self, lbl):
        return self.track.xs(lbl, level=1)

    def segments(self):
        for lbl in self.labels:
            yield self.get_segment(lbl)

    def show(self, ndims=2, **kwargs):

        import matplotlib
        matplotlib.rcParams['backend'] = 'Qt4Agg'
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        if ndims == 3:
            fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
        else:
            fig, axes = plt.subplots(1, 2)
        ax0, ax1 = axes
        for label in self.labels:
            if ndims == 3:
                ax0, ax1 = self.show_segment_3D(label, axes, **kwargs)
            else:
                ax0, ax1 = self.show_segment_2D(label, axes, **kwargs)
        return ax0, ax1

    def show_3D(self, **kwargs):
        return self.show(ndims=3, **kwargs)

    def show_segment_3D(self, label, axes=None, coords=('x', 'y', 'z')):

        import matplotlib
        matplotlib.rcParams['backend'] = 'Qt4Agg'
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        if axes is None:
            fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
        ax0, ax1 = axes
        segment = self.get_segment(label)

        xs = segment[coords[0]].values
        ys = segment[coords[1]].values
        zs = segment[coords[2]].values

        times = segment.index.get_level_values(0)
        ax0.plot(times, xs, ys)
        colors = plt.cm.jet(xs.size)
        ax1.plot(xs, ys, zs)
        ax1.scatter(xs, ys, zs, c=colors)
        ax0.set_xlabel('Time (min)')
        ax0.set_ylabel(u'x position (µm)')
        ax0.set_zlabel(u'y position (µm)')
        ax1.set_xlabel(u'x position (µm)')
        ax1.set_ylabel(u'y position (µm)')
        ax1.set_zlabel(u'z position (µm)')

        return ax0, ax1

    def show_segment_2D(self, label, axes=None, coords=('x', 'y')):

        import matplotlib
        matplotlib.rcParams['backend'] = 'Qt4Agg'
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        if axes is None:
            fig, axes = plt.subplots(1, 2)
        ax0, ax1 = axes
        segment = self.get_segment(label)
        xs = segment[coords[0]].values
        ys = segment[coords[1]].values

        times = segment.index
        ax0.plot(times, xs)
        colors = plt.cm.jet(xs.size)
        ax1.plot(xs, ys)
        ax1.scatter(xs, ys,
                    c=colors)
        ax0.set_xlabel('Time (min)')
        ax0.set_ylabel(u'x position (µm)')
        ax1.set_xlabel(u'x position (µm)')
        ax1.set_ylabel(u'y position (µm)')
        return ax0, ax1

    def do_pca(self, df=None, ndims=3,
               coords=['x', 'y', 'z'], suffix='_pca'):

        import matplotlib.pyplot as plt

        if not df:
            df = self.track
        self.pca = PCA()
        pca_coords = [c + suffix for c in coords]
        if ndims == 2:
            coords = coords[:2]
            pca_coords = pca_coords[:2]

        rotated = self.pca.fit_transform(df[coords])
        for n, coord in enumerate(pca_coords):
            df[coord] = rotated[:, n]
        return df

    @property
    def colors(self):
        '''
        Returns a DataFrame indexed like `self.track` with a
        color for each unique label
        '''
        clrs = self.track.index.get_level_values(
            'label').values.astype(np.float)
        clrs /= clrs.max()
        clrs = pd.DataFrame(plt.cm.spectral(clrs),
                            index=self.track.index,
                            columns=('R', 'G', 'B', 'A'))
        return clrs

    @property
    def label_colors(self):
        '''dictionary with labels as key and a single RGBA
        quadruplets for each label
        '''
        return {label: tuple(self.colors.xs(label, level='label').iloc[0].values)
                for label in self.labels}


def relabel_fromzero(df, level, inplace=False):

    old_lbls = df.index.get_level_values(level)
    nu_lbls = old_lbls.values.astype(np.uint16).copy()
    for n, uv in enumerate(old_lbls.unique()):
        nu_lbls[old_lbls == uv] = n
    if not inplace:
        df = df.copy()
    df['new_label'] = nu_lbls
    df.set_index('new_label', append=True, inplace=True)
    df.reset_index(level, drop=True, inplace=True)
    names = list(df.index.names)
    names[names.index('new_label')] = level
    df.index.set_names(names, inplace=True)
    return df


def _predict_coordinate(segment, coord, times, t1, sigma=10., **kwargs):

    times = np.atleast_2d(times).T
    prev = segment[coord]
    nugget = (sigma / (prev + sigma)) ** 2
    gp = GaussianProcess(nugget=nugget, **kwargs)
    gp.fit(times, prev)
    return gp.predict(t1, eval_MSE=True)
