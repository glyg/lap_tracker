#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd

from .lap_tracking import LAPTracker


DEFAULT_PARAMS = {'n_part':5,
                  'n_times':100,
                  'noise':1e-8,
                  'p_disapear':1e-8,
                  'sampling':10,
                  'max_disp':0.1,
                  'window_gap':10,
                  'gp_corr':'squared_exponential',
                  'gp_regr':'quadratic',
                  'gp_theta0':0.1}

def test_tracker(params=DEFAULT_PARAMS):

    n_part = params['n_part']
    n_times = params['n_times']
    noise = params['noise']
    p_disapear = params['p_disapear']
    sampling = params['sampling']
    data, teststore = make_data(n_part, n_times, noise,
                                p_disapear, sampling)

    test_track = LAPTracker(data, teststore, params)
    test_track.dist_function = lambda x:x
    test_track.get_track(predict=False)
    # test_track.reverse_track()
    # test_track.get_track(predict=True)
    # test_track.reverse_track()
    # test_track.get_track(predict=True)
    test_track.close_merge_split()
    scores = {}
    for label in test_track.labels:
        segment = test_track.get_segment(label)
        good = segment['good_lbls']
        bc = np.bincount(good.values.astype(np.int))
        scores[label] = bc.max()/bc.sum() * 100
    global_score = np.mean([score for score in scores.values()])
    print('Global: %.3f' % global_score)
    print('Number of individual trajectories: %i'
          % test_track.labels.shape[0])
    return scores, test_track


def make_data(n_part=5, n_times=100, noise=1e-10,
              p_disapear=1e-10, sampling=10):
    '''Creates a DataFrame containing simulated trajectories

    Parameters:
    ===========
    n_part: int, the number of trajectories
    n_times: int, the number of time points
    noise: float, the typical position noise
    p_disapear: float, the probability that a particle disapears in one frame
    sampling: int, the typical density of points w/r to trajectories
        variations. This corresponds typicaly to the number of points
        for one period of an oscillating function.

    Returns:
    ========
    raw: a pd.DataFrame correctly indexed for tracking.
        The column 'good_lbls' contains the original labels,
        before shuffeling
    testore: and pd.HDFStore object where `raw` is stored
    '''
    np.random.seed(42)
    times = np.arange(n_times)
    phases = np.random.random(n_part) * 2 * np.pi
    initial_positions = np.random.random((n_part, 3))
    labels = range(n_part) * n_times
    time_stamps = np.array([range(n_times)] * n_part).T.flatten()
    tuples = [(t, lbl) for t, lbl in zip(time_stamps, labels)]
    index = pd.MultiIndex.from_tuples(tuples, names=('t', 'label'))
    all_pos = np.zeros((n_times * n_part, 3))
    for n in range(n_part):
        phase = phases[n]
        pos0 = initial_positions[n, :]
        pos_err = np.random.normal(0, noise, (n_times, 3))
        all_pos[n::n_part] = (pos0
                              + positions(times, phase, sampling)
                              + pos_err)
    raw = pd.DataFrame(all_pos, index, columns=('x', 'y', 'z'))
    disapear = np.random.binomial(n_part, p_disapear, n_times * n_part)
    disapear = np.where(disapear == 1)[::-1][0]
    if disapear.size > 0:
        raw = raw.drop(raw.index[disapear])
    raw['good_lbls'] = raw.index.get_level_values(1)
    grouped = raw.groupby(level='t')
    raw = grouped.apply(shuffle)

    teststore = pd.HDFStore('test.h5')
    teststore['raw'] = raw
    teststore.close()
    return raw, teststore

def shuffle(df):
    '''
    shuffles the input dataframe and returns it
    '''
    values = df.values
    np.random.shuffle(values)
    df = pd.DataFrame(values, index=df.index,
                      columns=df.columns)
    return df


def positions(times, phase, sampling=5):
    '''
    computes a swirly trajectory
    '''
    sampling *= 2. * np.pi
    xs = times * np.cos(times / sampling + phase)
    ys = np.sin(times / sampling - phase)**2
    zs = times / 10.
    xs /= xs.ptp()
    ys /= ys.ptp()
    zs /= zs.ptp()

    return np.array([xs, ys - ys[0], zs]).T
