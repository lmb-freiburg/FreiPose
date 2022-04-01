from __future__ import print_function, unicode_literals
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def _calc_fig_shape(num_samples):
    """ Calculates the size of a N, M grid so that num_samples can be accomodated within.
        When N, M correspond to rows, columns then it favors a landscapeish output.
    """
    M = int(np.ceil(np.sqrt(num_samples)))
    for i in range(1, M + 1):
        if i * M >= num_samples:
            N = i
            break

    return N, M


def plt_figure(num_plots, fig_id=None, axis_titles=None, is_3d_axis=None):
    """ Utility to quicker set up matplotlib figure in a style I commonly use. """
    assert num_plots > 0, 'Num plot has to be greater equal 1.'
    assert num_plots <= 9, 'Cant deal with so many plots.'
    if axis_titles is not None:
        assert len(axis_titles) == num_plots, 'There have to be as many titles as plots.'
    if is_3d_axis is not None:
        assert type(is_3d_axis) == list, 'is_3d_axis be a list.'
        assert [type(x) == list for x in is_3d_axis], 'is_3d_axis be a list of ints.'
        assert (max(is_3d_axis) < num_plots) and (max(is_3d_axis) >= 0), 'is_3d_axis range mismatch.'

    fig = plt.figure(fig_id)
    N, M = _calc_fig_shape(num_plots)

    axes = list()
    for i in range(num_plots):
        ax_id = N*100+M*10+i+1
        if is_3d_axis is None or i not in is_3d_axis:
            axes.append(fig.add_subplot(ax_id))
        else:
            axes.append(fig.add_subplot(ax_id, projection='3d'))

        if axis_titles is not None:
            axes[-1].set_title(axis_titles[i])

    return plt, fig, axes


