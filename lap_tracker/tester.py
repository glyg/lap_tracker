# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd

from .lap_tracking import LAPTracker


DEFAULT_PARAMS = {'n_part': 5,
                  'n_times': 100,
                  'noise': 1e-8,
                  'p_disapear': 1e-8,
                  'sampling': 10,
                  'max_disp': 0.1,
                  'window_gap': 10,
                  'gp_corr': 'squared_exponential',
                  'gp_regr': 'quadratic',
                  'gp_theta0': 0.1}


def generate_split(size, noise, split_time):

    np.random.seed(0)
    ts = np.linspace(0.1, 1., size)
    x1 = ts + np.random.normal(0, noise, size)
    x2 = np.sin(ts) + np.random.normal(0, noise, size)

    y1 = np.cos(ts) + np.random.normal(0, noise, size)
    y2 = np.cos(ts) + noise + np.random.normal(0, noise, size)

    I1 = np.random.normal(1, noise, size)
    I2 = np.random.normal(1, noise, size)

    x2[:split_time] = np.nan
    y2[:split_time] = np.nan
    I1[:split_time] += I2[:split_time]
    I2[:split_time] = np.nan

    xs = np.vstack((x1, x2)).T.flatten()
    ys = np.vstack((y1, y2)).T.flatten()
    Is = np.vstack((I1, I2)).T.flatten()

    data = np.vstack((xs, ys, Is)).T
    t_stamp, label = np.mgrid[0:size, 0:2]

    index = pd.MultiIndex.from_tuples([(t, l) for t, l
                                       in zip(t_stamp.ravel(),
                                              label.ravel())],
                                      names=('t', 'label'))

    t_split = pd.DataFrame(data, index=index, columns=('x', 'y', 'I'))
    t_split = t_split.dropna()
    t_split = LAPTracker(t_split)
    t_split.ndims = 2
    t_split.max_disp = 2. / size

    return t_split


def generate_gap(size, noise, gap_start,
                 gap_stop, traj_shift=1.):

    np.random.seed(0)
    ts = np.linspace(0.1, 1., size)
    x1 = np.cos(ts) + np.random.normal(0, noise, size) + traj_shift
    x2 = np.sin(ts) + np.random.normal(0, noise, size)

    y1 = np.cos(ts) + np.random.normal(0, noise, size)
    y2 = np.cos(ts) + noise + np.random.normal(0, noise, size)

    I1 = np.random.normal(1, noise, size)
    I2 = np.random.normal(1, noise, size)

    x2[gap_start: gap_stop] = np.nan
    y2[gap_start: gap_stop] = np.nan
    I2[gap_start: gap_stop] = np.nan

    xs = np.vstack((x1, x2)).T.flatten()
    ys = np.vstack((y1, y2)).T.flatten()
    Is = np.vstack((I1, I2)).T.flatten()

    data = np.vstack((xs, ys, Is)).T

    t_stamp, label = np.mgrid[0:size, 0:2]
    label[gap_stop:, 1] = 2
    index = pd.MultiIndex.from_tuples([(t, l) for t, l
                                       in zip(t_stamp.ravel(),
                                              label.ravel())],
                                      names=('t', 'label'))
    t_gap = pd.DataFrame(data, index=index, columns=('x', 'y', 'I'))
    t_gap = t_gap.dropna()
    t_gap = LAPTracker(t_gap)
    t_gap.ndims = 2
    t_gap.dist_function = lambda x: x
    t_gap.max_disp = 2. / size
    return t_gap


def generate_merge(size, noise, merge_time):

    np.random.seed(0)
    ts = np.linspace(0.1, 1., size)
    x1 = ts + np.random.normal(0, noise, size)
    x2 = np.sin(ts) + np.random.normal(0, noise, size)

    y1 = np.cos(ts) + np.random.normal(0, noise, size)
    y2 = np.cos(ts) + noise + np.random.normal(0, noise, size)

    I1 = np.random.normal(1, noise, size)
    I2 = np.random.normal(1, noise, size)

    x2[: merge_time] = np.nan
    y2[: merge_time] = np.nan
    I1[: merge_time] += I2[: merge_time]
    I2[: merge_time] = np.nan

    xs = np.vstack((x1, x2)).T.flatten()
    ys = np.vstack((y1, y2)).T.flatten()
    Is = np.vstack((I1, I2)).T.flatten()

    data = np.vstack((xs, ys, Is)).T
    data = data[::-1]
    t_stamp, label = np.mgrid[0:size, 0:2]
    index = pd.MultiIndex.from_tuples([(t, l) for t, l
                                       in zip(t_stamp.ravel(),
                                              label.ravel())],
                                      names=('t', 'label'))
    t_merge = pd.DataFrame(data, index=index, columns=('x', 'y', 'I'))
    t_merge = t_merge.dropna()
    t_merge = LAPTracker(t_merge)
    t_merge.ndims = 2
    t_merge.max_disp = 2. / size
    return t_merge


def generate_merge_split(size, noise, merge_time):

    t_split0 = generate_split(size, noise, merge_time)
    t_merge_split = generate_merge(size, noise, merge_time)

    t = t_merge_split.track.index.get_level_values('t')
    t += t_split0.track.index.get_level_values('t')[-1] + 1

    t_merge_split.track['new_t'] = t
    t_merge_split.track.set_index('new_t', drop=True,
                                  inplace=True, append=True)
    t_merge_split.track.reset_index(level='t', drop=True, inplace=True)
    t_merge_split.track = t_merge_split.track.swaplevel('new_t', 'label')

    tmp_label = t_merge_split.track.index.get_level_values('label').values
    tmp_label[tmp_label == 0] = 2
    tmp_label[tmp_label == 1] = 3
    #tmp_label[tmp_label == -1] = 1
    t_merge_split.track['tmp_label'] = tmp_label
    t_merge_split.track.set_index('tmp_label', drop=True,
                                  inplace=True, append=True)
    t_merge_split.track.reset_index(level='label', drop=True, inplace=True)
    t_merge_split.track.index.set_names(['t', 'label'])

    t_merge_split.track.index.set_names(['t', 'label'])
    t_merge_split.track = pd.concat([t_split0.track, t_merge_split.track])
    return t_merge_split


def test_tracker(params=DEFAULT_PARAMS):

    n_part = params['n_part']
    n_times = params['n_times']
    noise = params['noise']
    p_disapear = params['p_disapear']
    sampling = params['sampling']
    data, teststore = make_data(n_part, n_times, noise,
                                p_disapear, sampling)

    test_track = LAPTracker(data, teststore, params=params)
    test_track.dist_function = lambda x: x

    ## Straight 1
    test_track.get_track(predict=False)
    print('''Number of segments after first pass: %d'''
          % test_track.labels.size)

    ## Reversed
    test_track.reverse_track()
    test_track.get_track(predict=False)
    print('''Number of segments after 2nd pass: %d'''
          % test_track.labels.size)
    test_track.reverse_track()

    ### Straight 2
    test_track.get_track(predict=False)
    print('''Number of segments after 3rd pass: %d'''
          % test_track.labels.size)

    # test_track.close_merge_split(gap_close_only=True)
    # print('''Number of segments after gap close: %d'''
    #       % test_track.labels.size)
    # test_track.close_merge_split(gap_close_only=False)
    # print('''Number of segments after merge/split: %d'''
    #       % test_track.labels.size)
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
    time_stamps, labels = np.mgrid[:n_times, :n_part]
    index = pd.MultiIndex.from_arrays([time_stamps.flatten(),
                                       labels.flatten()],
                                      names=('t', 'label'))
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
