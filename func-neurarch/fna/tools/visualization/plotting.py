# noinspection ProblematicWhitespace
"""
========================================================================================================================
Plotting Package
========================================================================================================================
(incomplete documentation)

Provides all the relevant classes, methods and functions for plotting routines

Classes:
----------
    SpikePlots - wrapper class for all plotting routines applied to population or network_architect spiking data
    AnalogSignalPlots - wrapper class for all plotting routines associated with continuous, analog recorded data
    TopologyPlots - wrapper class for all plotting routines related to network_architect structure and connectivity

Functions:
----------

========================================================================================================================
Copyright (C) 2018  Renato Duarte, Barna Zajzon

Neural Mircocircuit Simulation and Analysis Toolkit is free software;
you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation;
either version 2 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

The GNU General Public License does not permit this software to be
redistributed in proprietary programs.

"""
import itertools
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as pl, axes, mlab
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes, mark_inset
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats as st
import sklearn.decomposition as sk
from sklearn.cluster import SpectralBiclustering
import importlib

nest = None if importlib.util.find_spec("nest") is None else importlib.import_module("nest")
try:
    from nest import topology as tp
except:
    pass

# internal imports
from fna.tools import utils
from fna.tools import signals, network_architect as net
from fna.tools.analysis import metrics
from fna.tools.network_architect.connectivity import spectral_radius
from fna.tools.network_architect.topology import get_center
import fna.tools.visualization as viz
from fna.tools.visualization.helper import (get_cmap, isi_analysis_histogram_axes, summary_statistics,
                                            mark_epochs, check_axis, fig_output)
from fna.tools import check_dependency

logger = utils.logger.get_logger(__name__)

# check optional dependencies
# has_networkx = check_dependency('networkx')
has_mayavi = check_dependency('mayavi.mlab')
if has_mayavi:
    pass


# TODO - incorporate in SpikeList
class SpikePlots(object):
    """
    Wrapper object with all the methods and functions necessary to visualize spiking
    activity from a simple dot display to more visually appealing rasters,
    as well as histograms of the most relevant statistical descriptors and so on..
    """

    def __init__(self, spikelist, start=None, stop=None, N=None):
        """
        Initialize SpikePlot object
        :param spikelist: SpikeList object, sliced to match the (start, stop) interval
        :param start: [float] start time for the display (if None, range is taken from data)
        :param stop: [float] stop time (if None, range is taken from data)
        """
        if not isinstance(spikelist, signals.spikes.SpikeList):
            raise Exception("Error, argument should be a SpikeList object")

        if start is None:
            self.start = spikelist.t_start
        else:
            self.start = start
        if stop is None:
            self.stop = spikelist.t_stop
        else:
            self.stop = stop
        if N is None:
            self.N = len(spikelist.id_list)

        self.spikelist = spikelist.time_slice(self.start, self.stop)

    def dot_display(self, gids_colors=None, with_rate=True, dt=1.0, display=True, ax=None, save=False,
                    default_color='b', fig=None, **kwargs):
        """
        Simplest case, dot display
        :param gids_colors: [list] if some ids should be highlighted in a different color, this should be specified by
        providing a list of (gids, color) pairs, where gids [numpy.ndarray] contains the ids and color is the
        corresponding color for those gids. If None, no ids are differentiated
        :param with_rate: [bool] - whether to display psth or not
        :param dt: [float] - delta t for the psth
        :param display: [bool] - display the figure
        :param ax: [axes handle] - axes on which to display the figure
        :param save: [bool] - save the figure
        :param default_color: [char] default color if no ids are differentiated
		:param fig: [matplotlib.figure]
        :param kwargs: [key=value pairs] axes properties
        """
        if (ax is not None) and (not isinstance(ax, list)) and (not isinstance(ax, mpl.axes.Axes)):
            raise ValueError('ax must be matplotlib.axes.Axes instance.')
        elif (ax is not None) and (isinstance(ax, list)):
            for axis_ax in ax:
                if not isinstance(axis_ax, mpl.axes.Axes):
                    raise ValueError('ax must be matplotlib.axes.Axes instance.')

        if ax is None:
            fig = pl.figure()
            if 'suptitle' in kwargs:
                fig.suptitle(kwargs['suptitle'])
                kwargs.pop('suptitle')
            if with_rate:
                ax1 = pl.subplot2grid((30, 1), (0, 0), rowspan=23, colspan=1)
                ax2 = pl.subplot2grid((30, 1), (24, 0), rowspan=5, colspan=1, sharex=ax1)
                ax2.set(xlabel='Time [ms]', ylabel='Rate')
                ax1.set(ylabel='Neuron')
            else:
                ax1 = fig.add_subplot(111)
        else:
            if with_rate:
                assert isinstance(ax, list), "Incompatible properties... (with_rate requires two axes provided or None)"
                ax1 = ax[0]
                ax2 = ax[1]
            else:
                ax1 = ax

        if 'suptitle' in kwargs and fig is not None:
            fig.suptitle(kwargs['suptitle'])
            kwargs.pop('suptitle')

        # extract properties from kwargs and divide them into axes properties and others
        ax_props = {k: v for k, v in kwargs.items() if k in ax1.properties()}
        pl_props = {k: v for k, v in kwargs.items() if k not in ax1.properties()}  # TODO: improve

        if gids_colors is None:
            times = self.spikelist.raw_data()[:, 0]
            neurons = self.spikelist.raw_data()[:, 1]
            ax1.plot(times, neurons, '.', color=default_color)
            ax1.set(ylim=[np.min(self.spikelist.id_list), np.max(self.spikelist.id_list)], xlim=[self.start, self.stop])
        else:
            assert isinstance(gids_colors, list), "gids_colors should be a list of (gids[list], color) pairs"
            ax_min_y = np.max(self.spikelist.id_list)
            ax_max_y = np.min(self.spikelist.id_list)
            for gid_color_pair in gids_colors:
                gids, color = gid_color_pair
                assert isinstance(gids, np.ndarray), "Gids should be a numpy.ndarray"

                tt = self.spikelist.id_slice(gids)  # it's okay since slice always returns new object
                times = tt.raw_data()[:, 0]
                neurons = tt.raw_data()[:, 1]
                ax1.plot(times, neurons, '.', color=color)
                ax_max_y = max(ax_max_y, max(tt.id_list))
                ax_min_y = min(ax_min_y, min(tt.id_list))
            ax1.set(ylim=[ax_min_y, ax_max_y], xlim=[self.start, self.stop])

        if with_rate:
            global_rate = self.spikelist.firing_rate(dt, average=True)
            mean_rate 	= self.spikelist.firing_rate(10., average=True)
            max_rate    = max(global_rate) + 1
            min_rate    = min(global_rate) + 1
            if gids_colors is None:
                time = self.spikelist.time_axis(dt)[:-1]
                ax2.plot(time, global_rate, **pl_props)
            else:
                assert isinstance(gids_colors, list), "gids_colors should be a list of (gids[list], color) pairs"
                for gid_color_pair in gids_colors:
                    gids, color = gid_color_pair
                    assert isinstance(gids, np.ndarray), "Gids should be a numpy.ndarray"

                    tt = self.spikelist.id_slice(gids)  # it's okay since slice always returns new object
                    time = tt.time_axis(dt)[:-1]
                    rate = tt.firing_rate(dt, average=True)
                    ax2.plot(time, rate, color=color, linewidth=1.0, alpha=0.8)
                    max_rate = max(rate) if max(rate) > max_rate else max_rate
            ax2.plot(self.spikelist.time_axis(10.)[:-1], mean_rate, 'k', linewidth=1.5)
            ax2.set(ylim=[min_rate, max_rate], xlim=[self.start, self.stop])
        else:
            ax1.set(**ax_props)
        if save:
            assert isinstance(save, str), "Please provide filename to save figure"
            pl.savefig(save)

        if display:
            pl.show(block=False)

    @staticmethod
    def mark_events(ax, input_obj, start=None, stop=None):
        """
        Highlight stimuli presentation times in axis
        :param ax:
        :param input_obj:
        :param start:
        :param stop:
        :return:
        """
        if not isinstance(ax, mpl.axes.Axes):
            raise ValueError('ax must be matplotlib.axes.Axes instance.')
        if start is None:
            start = ax.get_xlim()[0]
        if stop is None:
            stop = ax.get_xlim()[1]

        color_map = viz.helper.get_cmap(input_obj.dimensions)
        y_range = np.arange(ax.get_ylim()[0], ax.get_ylim()[1], 1)

        for k in range(input_obj.dimensions):
            onsets = input_obj.onset_times[k]
            offsets = input_obj.offset_times[k]

            assert(len(onsets) == len(offsets)), "Incorrect input object parameters"

            for idx, on in enumerate(onsets):
                if start-500 < on < stop+500:
                    ax.fill_betweenx(y_range, on, offsets[idx], facecolor=color_map(k), alpha=0.3)

    def print_activity_report(self, results=None, label='', n_pairs=500):
        """
        Displays on screen a summary of the network_architect settings and main statistics
        :param label: Population name
        """
        tt = self.spikelist.time_slice(self.start, self.stop)

        if results is None:
            stats = {}
            stats.update(metrics.compute_isi_stats(tt, summary_only=True, display=True))
            stats.update(metrics.compute_spike_stats(tt, time_bin=1., summary_only=True, display=True))
            stats.update(metrics.compute_synchrony(tt, n_pairs=n_pairs, time_bin=1., tau=20.,
                                                   time_resolved=False, depth=1))
        else:
            stats = results

        print('\n###################################################################')
        print(' Activity recorded in [%s - %s] ms, from population %s ' % (str(self.start), str(self.stop), str(label)))
        print('###################################################################')
        print('Spiking Neurons: {0}/{1}'.format(str(len(np.nonzero(tt.mean_rates())[0])), str(self.N)))
        print('Average Firing Rate: %.2f / %.2f Hz' % (np.mean(np.array(tt.mean_rates())[np.nonzero(tt.mean_rates())[0]]),
                                                       np.mean(tt.mean_rates())))
        # print 'Average Firing Rate (normalized by N): %.2f Hz' % (np.mean(tt.mean_rates()) * len(tt.id_list)) / self.N
        print('Fano Factor: %.2f' % stats['ffs'][0])
        print('*********************************\n\tISI metrics:\n*********************************')
        if 'lvs' in list(stats.keys()):
            print(('\t- CV: %.2f / - LV: %.2f / - LVR: %.2f / - IR: %.2f' % (stats['cvs'][0], stats['lvs'][0],
                                                                            stats['lvRs'][0], stats['iR'][0])))
            print(('\t- CVlog: %.2f / - H: %.2f [bits/spike]' % (stats['cvs_log'][0], stats['ents'][0])))
            print(('\t- 5p: %.2f ms' % stats['isi_5p'][0]))
        else:
            print('\t- CV: %.2f' % np.mean(stats['cvs']))

        print('*********************************\n\tSynchrony metrics:\n*********************************')
        if 'ccs_pearson' in list(stats.keys()):
            print(('\t- Pearson CC [{0} pairs]: {1}'.format(str(n_pairs), stats['ccs_pearson'][0])))
            print(('\t- CC [{0} pairs]: {1}'.format(str(n_pairs), str(stats['ccs'][0]))))
            if 'd_vr' in list(stats.keys()) and isinstance(stats['d_vr'], float):
                print(('\t- van Rossum distance: {0}'.format(str(stats['d_vr']))))
            elif 'd_vr' in list(stats.keys()) and not isinstance(stats['d_vr'], float):
                print(('\t- van Rossum distance: {0}'.format(str(np.mean(stats['d_vr'])))))
            if 'd_vp' in list(stats.keys()) and isinstance(stats['d_vp'], float):
                print(('\t- Victor Purpura distance: {0}'.format(str(stats['d_vp']))))
            elif 'd_vp' in list(stats.keys()) and not isinstance(stats['d_vp'], float):
                print(('\t- Victor Purpura distance: {0}'.format(str(np.mean(stats['d_vp'])))))
            if 'SPIKE_distance' in list(stats.keys()) and isinstance(stats['SPIKE_distance'], float):
                print(('\t- SPIKE similarity: %.2f / - ISI distance: %.2f ' % (stats[
                                                                                  'SPIKE_distance'], stats['ISI_distance'])))
            elif 'SPIKE_distance' in list(stats.keys()) and not isinstance(stats['SPIKE_distance'], float):
                print(('\t- SPIKE similarity: %.2f / - ISI distance: %.2f' % (np.mean(stats['SPIKE_distance']),
                                                                             np.mean(stats['ISI_distance']))))
            if 'SPIKE_sync' in list(stats.keys()):
                print(('\t- SPIKE Synchronization: %.2f' % np.mean(stats['SPIKE_sync'])))
        elif 'ccs' in list(stats.keys()):
            print(('\t- Pearson CC [{0} pairs]: {1}'.format(str(n_pairs), np.mean(stats['ccs']))))


# TODO - incorporate in AnalogSignal
class AnalogSignalPlots(object):
    """
    Wrapper object for all plots pertaining to continuous signals
    """

    def __init__(self, analog_signal_list, start=None, stop=None):
        """
        Initialize AnalogSignalPlot object
        :param analog_signal_list: AnalogSignalList object
        :param start: [float] start time for the display (if None, range is taken from data)
        :param stop: [float] stop time (if None, range is taken from data)
        """
        if (not isinstance(analog_signal_list, signals.analog.AnalogSignalList)) and (not isinstance(analog_signal_list,
                                                                                                             signals.analog.AnalogSignal)):
            raise Exception("Error, argument should be an AnalogSignal or AnalogSignalList")

        self.signal_list = analog_signal_list

        if start is None:
            self.start = self.signal_list.t_start
        else:
            self.start = start
        if stop is None:
            self.stop = self.signal_list.t_stop
        else:
            self.stop = stop

    def plot(self, ax=None, display=True, save=False, **kwargs):
        """
        Simply plot the contents of the AnalogSignal
        :param ax: axis handle
        :param display: [bool]
        :param save: [bool]
        :param kwargs: extra key-word arguments - particularly important are the axis labels
        and the plot colors
        """
        fig, ax = viz.helper.check_axis(ax)

        # extract properties from kwargs and divide them into axes properties and others
        ax_props = {k: v for k, v in kwargs.items() if k in ax.properties()}
        pl_props = {k: v for k, v in kwargs.items() if k not in ax.properties()}  # TODO: improve
        tt = self.signal_list.time_slice(self.start, self.stop)

        if isinstance(self.signal_list, signals.analog.AnalogSignal):
            times = tt.time_axis()
            signal = tt.raw_data()
            ax.plot(times, signal, **pl_props)

        elif isinstance(self.signal_list, signals.analog.AnalogSignalList):
            ids = self.signal_list.raw_data()[:, 1]
            for n in np.unique(ids):
                tmp = tt.id_slice([n])
                signal = tmp.raw_data()[:, 0]
                times = tmp.time_axis()
                ax.plot(times, signal, **pl_props)

        ax.set(**ax_props)
        ax.set(xlim=[self.start, self.stop])

        if display:
            pl.show(block=False)
        if save:
            assert isinstance(save, str), "Please provide filename"
            pl.savefig(save)

    def plot_Vm(self, ax=None, with_spikes=True, v_reset=None, v_th=None, display=True, save=False, **kwargs):
        """
        Special function to plot the time course of the membrane potential with or without highlighting the spike times
        :param with_spikes: [bool]
        """
        fig, ax = viz.helper.check_axis(ax)

        ax.set_xlabel('Time [ms]')
        ax.set_ylabel(r'V_{m} [mV]')
        ax.set_xlim(self.start, self.stop)

        tt = self.signal_list.time_slice(self.start, self.stop)

        if isinstance(self.signal_list, signals.analog.AnalogSignalList):
            ids = self.signal_list.raw_data()[:, 1]
            for n in np.unique(ids):
                tmp = tt.id_slice([n])
                vm = tmp.raw_data()[:, 0]
                times = tmp.time_axis()
        elif isinstance(self.signal_list, signals.analog.AnalogSignal):
            times = tt.time_axis()
            vm = tt.raw_data()
        else:
            raise ValueError("times and vm not specified")

        ax_props = {k: v for k, v in kwargs.items() if k in ax.properties()}
        pl_props = {k: v for k, v in kwargs.items() if k not in ax.properties()}  # TODO: improve

        if len(vm) != len(times):
            times = times[:-1]

        ax.plot(times, vm, 'k', **pl_props)

        if with_spikes:
            assert (v_reset is not None) and (v_th is not None), "To mark the spike times, please provide the " \
                                                                 "v_reset and v_th values"
            idxs = vm.argsort()
            possible_spike_times = [t for t in idxs if (t < len(vm) - 1) and (vm[t + 1] == v_reset) and (vm[t] !=
                                                                                                         v_reset)]
            ax.vlines(times[possible_spike_times], v_th, 50., color='k', **pl_props)
            ax.set_ylim(min(vm) - 5., 10.)
        else:
            ax.set_ylim(min(vm) - 5., max(vm) + 5.)

        ax.set(**ax_props)

        if display:
            pl.show(block=False)
        if save:
            assert isinstance(save, str), "Please provide filename"
            pl.savefig(save)
        return ax


# ######################################################################################################################
def recurrence_plot(time_series, dt=1, ax=None, color='k', type='.', display=True, save=False, **kwargs):
    """
    Plot a general recurrence plot of a 1D time series
    :param save:
    :param display:
    :param type:
    :param color:
    :param dt:
    :param time_series:
    :param ax:
    :param kwargs:
    :return:
    """
    fig, ax = viz.helper.check_axis(ax)
    in_pl = []
    ax_props = {k: v for k, v in kwargs.items() if k in ax.properties() and k not in in_pl}
    pl_props = {k: v for k, v in kwargs.items() if k not in ax.properties() or k in in_pl}

    for ii, isi_val in enumerate(time_series):
        if ii < len(time_series) - int(dt):
            ax.plot(isi_val, time_series[ii + int(dt)], type, c=color, **pl_props)
    ax.set(**ax_props)
    ax.set_xlabel(r'$x(t)$')
    ax.set_ylabel(r'$x(t-{0})$'.format(str(dt)))

    if save:
        assert isinstance(save, str), "Please provide filename"
        pl.savefig(save+'recurrence.pdf')

    if display:
        pl.show(block=False)


def plot_from_dict(dictionary, ax, bar_width=0.2):
    """
    Make a bar plot from a dictionary k: v, with xlabel=k and height=v
    :param dictionary: input dictionary
    :param ax: axis to plot on
    :param bar_width:
    :return:
    """
    x_labels = list(dictionary.keys())
    y_values = list(dictionary.values())
    ax.bar(np.arange(len(x_labels)), y_values, width=bar_width)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('Freq.')
    ax.set_xlabel('Token')


def plot_matrix(matrix, labels=None, ax=None, save=False, display=True, data_label=None):
    """
    Plots a 2D matrix as an image
    :return:
    """
    fig, ax = viz.helper.check_axis(ax)
    if data_label is not None:
        fig.suptitle(data_label)

    if np.array_equal(matrix, matrix.astype(bool)):
        plt = ax.imshow(1 - matrix, interpolation='nearest', aspect='auto', extent=None,
                        cmap='gray')
    else:
        plt = ax.imshow(matrix, aspect='auto', interpolation='nearest')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="4%")
        pl.colorbar(plt, cax=cax)

    if labels is not None:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
    # ax.set_yticks(np.arange(matrix.shape[0]))
    viz.helper.fig_output(fig, display, save)
    return fig, ax


def plot_spectrogram(spec, t, f, ax):
    """
    Plot a simple spectrogram
    :param spec: spectrogram
    :param t: time axis
    :param f: sampling frequency
    :param ax: axis
    :return:
    """
    ax.pcolormesh(t, f, spec)
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlable('Time [sec]')


def plot_single_raster(times, ax, t_start=0, t_stop=1000, save=False, display=True):
    """
    Plot the spike times of a single SpikeTrain as a vertical line
    :param times:
    :param ax:
    :param t_start:
    :param t_stop:
    :param save: False or string with full path to store
    :param display: bool
    :return:
    """
    fig, ax = viz.helper.check_axis(ax)
    for tt in times:
        ax.vlines(tt, 0.5, 1.5, color='k', linewidth=2)
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel(r'$\mathrm{S}_{i}$')
        ax.set_xlim([t_start, t_stop])
        ax.set_ylim([0.5, 1.6])
        ax.set_yticks([])
        ax.set_yticklabels([])
    viz.helper.fig_output(fig, display=display, save=save)


def plot_isis(isis, ax=None, save=False, display=False, **kwargs):
    """
    Plot the distribution of inter-spike-intervals provided
    :param isis:
    :param ax:
    :param save:
    :param display:
    :param kwargs:
    :return:
    """
    fig, ax = viz.helper.check_axis(ax)
    # ax2 = inset_axes(ax, width="60%", height=1.5, loc=1)
    ax_props, plot_props = viz.helper.parse_plot_arguments(ax, **kwargs)

    ax.plot(np.arange(len(isis)), isis, '.', **plot_props)
    ax.set(**ax_props)
    # ax2.plot(list(range(len(inset['isi']))), inset['isi'], '.')
    viz.helper.fig_output(fig, display=display, save=save)


def plot_io_curve(inputs, outputs, ax=None, save=False, display=False, **kwargs):
    """
    Plot any i/o curve
    :param inputs:
    :param outputs:
    :param ax:
    :param save:
    :param display:
    :param kwargs:
    :return:
    """
    fig, ax = viz.helper.check_axis(ax)
    ax_props, plot_props = viz.helper.parse_plot_arguments(ax, **kwargs)
    ax.plot(inputs, outputs, **plot_props)
    ax.set(**ax_props)
    viz.helper.fig_output(fig, display=display, save=save)


def plot_spectral_radius(w, ax=None, display=True, save=False):
    """
    Plot the spectral radius of the connectivity matrix
    :param w: matrix
    :param ax: axis where to plot (if None a new figure is generated)
    :param save: path to the folder where to save the figure
    :param display: [bool]
    :return:
    """
    fig, ax = viz.helper.check_axis(ax)
    eigs = spectral_radius(w)
    ax.scatter(np.real(eigs), np.imag(eigs))
    ax.set_title(r"$\rho(W)=$"+"{0!s}".format(np.max(np.real(eigs))))
    ax.set_xlabel(r"$\mathrm{Re(W)}$")
    ax.set_ylabel(r"$\mathrm{Im(W)}$")

    viz.helper.fig_output(fig, display=display, save=save)


def plot_singleneuron_isis(isis, ax=None, save=False, display=False, **kwargs):
    """
    Plot ISI distribution for a single neuron
    :param ax:
    :param save:
    :param display:
    :param kwargs:
    :return:
    """
    if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)):
        raise ValueError('ax must be matplotlib.axes.Axes instance.')

    if ax is None:
        fig = pl.figure()
        if 'suptitle' in kwargs:
            fig.suptitle(kwargs['suptitle'])
            kwargs.pop('suptitle')
        else:
            ax = fig.add_subplot(111)
    else:
        fig, ax = viz.helper.check_axis(ax)

    if 'inset' in kwargs.keys():
        inset = kwargs['inset']
        kwargs.pop('inset')
        ax2 = inset_axes(ax, width="60%", height=1.5, loc=1)
    else:
        inset = None
    in_pl = []
    ax_props = {k: v for k, v in kwargs.items() if k in ax.properties() and k not in in_pl}
    pl_props = {k: v for k, v in kwargs.items() if k not in ax.properties() or k in in_pl}

    ax.plot(range(len(isis)), isis, '.', **pl_props)
    ax.set(**ax_props)

    if inset is not None:
        ax2.plot(range(len(inset['isi'])), inset['isi'], '.')
        inset.pop('isi')

    viz.helper.fig_output(fig, display=display, save=save)


def plot_discrete_space(state_matrix, data_label='', label_seq=None, metric=None, colormap='jet', display=True,
                        save=False):
    """
    Plots a discrete state-space
    :return:
    """
    if state_matrix.shape[0] > 3:
        metrics.dimensionality_reduction(state_matrix, data_label, labels=label_seq, metric=metric, standardize=False,
                                 plot=True, colormap=colormap, display=display, save=save)

    elif state_matrix.shape[0] == 2:
        cmap = viz.helper.get_cmap(len(np.unique(label_seq)), cmap=colormap)
        scatter_projections(state_matrix.T, label_seq, cmap=cmap, display=display, save=save)

    elif state_matrix.shape[0] == 3:
        cmap = viz.helper.get_cmap(len(np.unique(label_seq)), cmap=colormap)
        scatter_projections(state_matrix.T, label_seq, cmap=cmap, display=display, save=save)


def plot_trajectory(response_matrix, pca_fit_obj=None, label='', color='r', ax=None, display=True, save=False):
    """

    :param response_matrix: [np.array] matrix of continuous responses
    :param pca_fit_obj:
    :param label:
    :param color:
    :param ax:
    :param display:
    :param save:
    :return:
    """
    fig, ax = viz.helper.check_axis(ax)

    if pca_fit_obj is None:
        pca_fit_obj = sk.PCA(n_components=min(response_matrix.shape))
    if not hasattr(pca_fit_obj, "explained_variance_ratio_"):
        pca_fit_obj.fit(response_matrix.T)
    X = pca_fit_obj.transform(response_matrix.transpose())
    # print("Explained Variance (first 3 components): %s" % str(pca_fit_obj.explained_variance_ratio_))

    # ax.clear()
    ax.plot(X[:, 0], X[:, 1], X[:, 2], color=color, lw=2, label=label)
    # ax.set_title(label + r' - (3PCs) = {0}'.format(str(round(np.sum(pca_fit_obj.explained_variance_ratio_[:3]), 1))))
    #ax.grid()

    viz.helper.fig_output(fig, display=display, save=save)


def plot_spatial_connectivity(network, src_gid=None, kernel=None, mask=None, ax=None, color='k'):
    """
    Plot spatial connectivity properties to axes
    :return:
    """
    if src_gid is None:
        center_gids, center_pos = get_center(network)
        src_gid = np.random.choice(list(center_gids))

    if (ax is not None) and (not isinstance(ax, axes.Axes)):
        raise ValueError('ax must be matplotlib.axes.Axes instance.')
    assert mask is not None, "Mask must be provided"

    if kernel is not None and isinstance(kernel, dict):
        if 'gaussian' in kernel:
            sigma = kernel['gaussian']['sigma']
            for r in range(3):
                ax.add_patch(pl.Circle(src_gid, radius=(r + 1) * sigma, zorder=-1000,
                                       fc='none', ec=color, lw=3, ls='dashed'))
        else:
            raise ValueError("Kernel type cannot be plotted with this version of Topology")

    srcpos = np.array(tp.GetPosition([src_gid])[0])

    if 'anchor' in mask:
        offs = np.array(mask['anchor'])
    else:
        offs = np.array([0., 0.])

    if 'circular' in mask:
        r = mask['circular']['radius']
        ax.add_patch(pl.Circle(srcpos + offs, radius=r, zorder=-1000,
                               fc='none', ec=color, lw=3))
    elif 'doughnut' in mask:
        r_in = mask['doughnut']['inner_radius']
        r_out = mask['doughnut']['outer_radius']
        ax.add_patch(pl.Circle(srcpos + offs, radius=r_in, zorder=-1000,
                               fc='none', ec=color, lw=3))
        ax.add_patch(pl.Circle(srcpos + offs, radius=r_out, zorder=-1000,
                               fc='none', ec=color, lw=3))
    elif 'rectangular' in mask:
        ll = mask['rectangular']['lower_left']
        ur = mask['rectangular']['upper_right']
        ax.add_patch(pl.Rectangle(srcpos + ll + offs, ur[0] - ll[0], ur[1] - ll[1],
                                  zorder=-1000, fc='none', ec=color, lw=3))
    else:
        raise ValueError('Mask type cannot be plotted with this version of PyTopology.')

    pl.draw()


def plot_network_topology(network, colors=None, ax=None, dim=2, display=True, save=False, **kwargs):
    """
    Plot the network's spatial arrangement
    :return:
    """
    assert isinstance(network, net.Network)
    if colors is not None:
        assert len(colors) == len(network.populations), "Specify one color per population"
    else:
        cmap = viz.helper.get_cmap(len(network.populations), 'jet')
        colors = [cmap(i) for i in range(len(network.populations))]

    if (ax is not None) and (not isinstance(ax, axes.Axes)):
        raise ValueError('ax must be matplotlib.axes.Axes instance.')

    if ax is None and dim < 3:
        fig, ax = pl.subplots()
    elif ax is None and dim == 3:
        fig = pl.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax_props = {k: v for k, v in kwargs.items() if k in ax.properties()}
    plot_props = {k: v for k, v in kwargs.items() if k not in ax.properties()}
    # ax.set_title(network.name)
    ax.set(**ax_props)

    for c, p in zip(colors, network.populations.values()):
        assert p.topology, "Population %s has no topology" % str(p.name)
        positions = list(zip(*[tp.GetPosition([n])[0] for n in nest.GetLeaves(p.layer_gid)[0]]))

        if len(positions) < 3:
            ax.plot(positions[0], positions[1], 'o', color=c, label=p.name, **plot_props)
        else:
            ax.scatter(positions[0], positions[1], positions[2], depthshade=True, c=c, label=p.name,
                           **plot_props)
    pl.legend(loc=1)
    if display:
        pl.show(block=False)
    if save:
        assert isinstance(save, str), "Please provide filename"
        pl.savefig(save)


def plot_histogram(data, n_bins, norm=True, mark_mean=False, mark_median=False, ax=None, color='b', display=True,
                   save=False, **kwargs):
    """
    Default histogram plotting routine
    :param data: data to plot (list or np.array)
    :param n_bins: number of bins to use (int)
    :param norm: normalized or not (bool)
    :param mark_mean: add a vertical line annotating the mean (bool)
    :param mark_median: add a vertical line annotating the median (bool)
    :param ax: axis to plot on (if None a new figure will be created)
    :param color: histogram color
    :param display: show figure (bool)
    :param save: save figure (False or string with figure path)
    :return n, bins: binned data
    """
    data = np.array(data)
    if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)):
        raise ValueError('ax must be matplotlib.axes.Axes instance.')

    if ax is None:
        fig = pl.figure()
        ax = fig.add_subplot(111)

    if 'suptitle' in kwargs:
        fig.suptitle(kwargs['suptitle'])
        kwargs.pop('suptitle')

    # extract properties from kwargs and divide them into axes properties and others
    in_pl = ['label', 'alpha', 'orientation']
    ax_props = {k: v for k, v in kwargs.items() if k in ax.properties() and k not in in_pl}
    pl_props = {k: v for k, v in kwargs.items() if k not in ax.properties() or k in in_pl}

    # if len(tmpa) > 1:
    # 	tmp = list(itertools.chain(*tmpa))
    data = data[data != 0]
    if np.any(data[np.isnan(data)]):
        print("Removing NaN")
        data = data[~np.isnan(data)]
    if np.any(data[np.isinf(data)]):
        print("Removing inf")
        data = data[~np.isinf(data)]

    n = 0
    bins = 0
    if norm and list(data):
        weights = np.ones_like(data) / float(len(data))
        n, bins, patches = ax.hist(data, n_bins, weights=weights, **pl_props)  # histtype='stepfilled', alpha=0.8)
        pl.setp(patches, 'facecolor', color)
    elif list(data):
        n, bins, patches = ax.hist(data, n_bins, **pl_props)
        pl.setp(patches, 'facecolor', color)

    if 'label' in list(pl_props.keys()):
        pl.legend()

    if mark_mean:
        ax.axvline(data.mean(), color=color, linestyle='dashed')
    if mark_median:
        ax.axvline(np.median(data), color=color, linestyle='dashed')

    ax.set(**ax_props)

    if save:
        assert isinstance(save, str), "Please provide filename"
        pl.savefig(save)

    if display:
        pl.show(block=False)

    return n, bins


def violin_plot(ax, data, pos, location=-1, color='y'):
    """
    Default violin plot routine
    :param ax:
    :param data:
    :param pos:
    :param location: location on the axis (-1 left,1 right or 0 both)
    :param color:
    :return:
    """
    dist = max(pos)-min(pos)
    w = min(0.15*max(dist, 1.0), 0.5)
    for d, p, c in zip(data, pos, color):
        k = st.gaussian_kde(d)     #calculates the kernel density
        m = k.dataset.min()     #lower bound of violin
        M = k.dataset.max()     #upper bound of violin
        x = np.arange(m, M, (M-m)/100.) # support for violin
        v = k.evaluate(x) #violin profile (density curve)
        v = v/v.max() * w #scaling the violin to the available space
        if location:
            ax.fill_betweenx(x, p, (location*v)+p, facecolor=c, alpha=0.3)
        else:
            ax.fill_betweenx(x, p, v + p, facecolor=c, alpha=0.3)
            ax.fill_betweenx(x, p, -v + p, facecolor=c, alpha=0.3)


def box_plot(ax, data, pos):
    """
    creates one or a set of boxplots on the axis provided
    :param ax: axis handle
    :param data: list of data points
    :param pos: list of x positions
    :return:
    """
    ax.boxplot(data, notch=1, positions=pos, vert=1, sym='')


def plot_histograms(ax_list, data_list, n_bins, args_list=None, colors=None, cmap='hsv', kde=False, display=True,
                    save=False):
    """

    :param ax_list:
    :param data_list:
    :param n_bins:
    :param args_list:
    :param cmap:
    :return:
    """
    assert(len(ax_list) == len(data_list)), "Data dimension mismatch"
    if colors is None:
        cc = get_cmap(len(ax_list), cmap)
        colors = [cc(ii) for ii in range(len(ax_list))]
    counter = list(range(len(ax_list)))
    for ax, data, c in zip(ax_list, data_list, counter):
        n, bins = plot_histogram(data, n_bins[c], ax=ax, color=colors[c], display=False,
                                 **{'histtype': 'stepfilled', 'alpha': 0.6})
        if kde:
            approximate_pdf_isi = st.kde.gaussian_kde(data)
            x = np.linspace(np.min(data), np.max(data), n_bins[c])
            y = approximate_pdf_isi(x)
            y /= np.sum(y)
            ax.plot(x, y, color=colors[c], lw=2)
        if args_list is not None:
            ax.set(**args_list[c])
        ax.set_ylim([0., np.max(n)])
    fig = pl.gcf()
    viz.helper.fig_output(fig, display=display, save=save)


def plot_state_analysis(parameter_set, results, summary_only=False, start=None, stop=None, display=True, save=False):
    """
    Plots spiking and analog activity. Spiking activity includes spike raster plot and other statistics (mean firing
    rate and CV-, FF-, and CCS histograms). If analog activity is recorded, the mean V_m is plotted, along with the
     mean E and mean I currents.

    :param parameter_set: ParameterSet object
    :param results: dictionary with the activity statistics
    :param summary_only: plot only raster plot or everything
    :param start: start of time window for the plots
    :param stop: end of time window for the plots
    :param display: show plots
    :param save: save plots
    :return:
    """
    fig2 = []
    fig3 = []
    fig1 = pl.figure()
    fig1.suptitle(r'Population {0} - Global Activity $[{1}, {2}]$'.format(
        str(parameter_set.kernel_pars.data_prefix + results[
            'metadata']['population_name']), str(start), str(stop)))
    if bool(results['analog_activity']):
        ax1 = pl.subplot2grid((23, 1), loc=(0, 0), rowspan=11, colspan=1)
        ax2 = pl.subplot2grid((23, 1), loc=(11, 0), rowspan=3, colspan=1, sharex=ax1)
        ax3 = pl.subplot2grid((23, 1), loc=(17, 0), rowspan=3, colspan=1, sharex=ax1)
        ax4 = pl.subplot2grid((23, 1), loc=(20, 0), rowspan=3, colspan=1, sharex=ax1)
    else:
        ax1 = pl.subplot2grid((25, 1), loc=(0, 0), rowspan=20, colspan=1)
        ax2 = pl.subplot2grid((25, 1), loc=(20, 0), rowspan=5, colspan=1, sharex=ax1)

    colors = ['b', 'r', 'Orange', 'gray', 'g'] # color sequence for the different populations (TODO automate)

    if bool(results['spiking_activity']):
        pop_names = list(results['spiking_activity'].keys())

        # make spike raster plot
        spiking_activity = results['metadata']['spike_list']
        rp = SpikePlots(spiking_activity, start, stop)
        if display:
            rp.print_activity_report(label=results['metadata']['population_name'],
                                     results=results['spiking_activity'][results['metadata']['population_name']])
        plot_props = {'xlabel': 'Time [ms]', 'ylabel': 'Neuron', 'color': 'b', 'linewidth': 1.0,
                      'linestyle': '-'}
        if len(pop_names) > 1 or 'sub_population_names' in list(results['metadata'].keys()):
            # different color for each subpopulation
            gids = results['metadata']['sub_population_gids']
            rp.dot_display(gids_colors=[(gids_, colors[idx]) for idx, gids_ in enumerate(gids)],
                           ax=[ax1, ax2], with_rate=True, display=False, save=False, **plot_props)
        else:
            # single global color
            rp.dot_display(ax=[ax1, ax2], with_rate=True, display=False, save=False, **plot_props)
        ax2.set_ylabel(r'$\mathrm{\bar{r}(t)} \mathrm{[sps]}$')
        ax1.set_ylabel(r'$\mathrm{Neuron}$')

        # plot other spiking activity related histograms
        if not summary_only:
            fig2 = pl.figure()
            fig2.suptitle(r'Population {0} - Spiking Statistics $[{1}, {2}]$'.format(
                str(parameter_set.kernel_pars.data_prefix + results['metadata']['population_name']),
                str(start), str(stop)))
            ax21 = pl.subplot2grid((9, 14), loc=(0, 0), rowspan=4, colspan=4)
            ax22 = pl.subplot2grid((9, 14), loc=(0, 5), rowspan=4, colspan=4)
            ax23 = pl.subplot2grid((9, 14), loc=(0, 10), rowspan=4, colspan=4)
            ax24 = pl.subplot2grid((9, 14), loc=(5, 3), rowspan=4, colspan=4)
            ax25 = pl.subplot2grid((9, 14), loc=(5, 8), rowspan=4, colspan=4)

            for idx, name in enumerate(pop_names):
                plot_props = {'xlabel': 'Rates', 'ylabel': 'Count', 'histtype': 'stepfilled', 'alpha': 0.4}

                plot_histogram(results['spiking_activity'][name]['mean_rates'], n_bins=100, norm=True, ax=ax21, color=colors[idx],
                               display=False, save=False, **plot_props)
                plot_props.update({'xlabel': 'ISI'})  # , 'yscale': 'log'}) #, 'xscale': 'log'})##
                ax22.set_yscale('log')
                plot_histogram(results['spiking_activity'][name]['isi'], n_bins=100, norm=True, ax=ax22, color=colors[idx],
                               display=False, save=False, **plot_props)
                plot_props['xlabel'] = 'CC'
                tmp = np.array(results['spiking_activity'][name]['ccs'])
                ccs = tmp[~np.isnan(tmp)] #tmp
                plot_histogram(ccs, n_bins=100, norm=True, ax=ax23, color=colors[idx],
                               display=False, save=False, **plot_props)

                plot_props['xlabel'] = 'FF'
                tmp = np.array(results['spiking_activity'][name]['ffs'])
                ffs = tmp[~np.isnan(tmp)]
                plot_histogram(ffs, n_bins=100, norm=True, ax=ax24, color=colors[idx],
                               display=False, save=False, **plot_props)

                plot_props['xlabel'] = '$CV_{ISI}$'
                tmp = np.array(results['spiking_activity'][name]['cvs'])
                cvs = tmp[~np.isnan(tmp)]
                plot_histogram(cvs, n_bins=100, norm=True, ax=ax25, color=colors[idx],
                               display=False, save=False, **plot_props)

    # plot analog activity statistics
    if bool(results['analog_activity']):
        pop_names = list(results['analog_activity'].keys())

        for idx, name in enumerate(pop_names):
            if len(results['analog_activity'][name]['recorded_neurons']) > 1:
                fig3 = pl.figure()
                fig3.suptitle(r'Population {0} - Analog Signal Statistics [${1}, {2}$]'.format(
                    str(parameter_set.kernel_pars.data_prefix + results[
                        'metadata']['population_name']), str(start), str(stop)))
                ax31 = pl.subplot2grid((6, 3), loc=(2, 0), rowspan=3, colspan=1)
                ax32 = pl.subplot2grid((6, 3), loc=(2, 1), rowspan=3, colspan=1)
                ax33 = pl.subplot2grid((6, 3), loc=(2, 2), rowspan=3, colspan=1)

                plot_props = {'xlabel': r'$\langle V_{m} \rangle$', 'ylabel': 'Count', 'histtype': 'stepfilled',
                              'alpha': 0.8}
                plot_histogram(results['analog_activity'][name]['mean_V_m'], n_bins=20, norm=True, ax=ax31,
                               color=colors[idx],
                               display=False, save=False, **plot_props)
                plot_props.update({'xlabel': r'$\langle I_{Syn}^{Total} \rangle$'}) #, 'label': r'\langle I_{Exc}
                # \rangle'})
                plot_histogram(results['analog_activity'][name]['mean_I_ex'], n_bins=20, norm=True, ax=ax32, color='b',
                               display=False, save=False, **plot_props)
                #plot_props.update({'label': r'\langle I_{Inh} \rangle'})
                plot_histogram(results['analog_activity'][name]['mean_I_in'], n_bins=20, norm=True, ax=ax32, color='r',
                               display=False, save=False, **plot_props)
                #plot_props.update({'label': r'\langle I_{Total} \rangle'})
                plot_histogram(np.array(results['analog_activity'][name]['mean_I_in']) + np.array(results[
                                                                                                      'analog_activity'][name]['mean_I_ex']), n_bins=20, norm=True, ax=ax32, color='gray',
                               display=False, save=False, **plot_props)
                plot_props.update({'xlabel': r'$CC_{I_{E}/I_{I}}$'})
                #plot_props.pop('label')
                plot_histogram(results['analog_activity'][name]['EI_CC'], n_bins=20, norm=True, ax=ax33, color=colors[
                    idx], display=False, save=False, **plot_props)
            elif results['analog_activity'][name]['recorded_neurons']:
                pop_idx = parameter_set.net_pars.pop_names.index(name)
                ###
                times = results['analog_activity'][name]['time_axis']
                vm = results['analog_activity'][name]['single_Vm']
                idx = results['analog_activity'][name]['single_idx']

                if len(vm) != len(times):
                    times = times[:-1]

                ax4.plot(times, vm, 'k', lw=1)
                idxs = vm.argsort()
                if 'V_reset' in list(parameter_set.net_pars.neuron_pars[pop_idx].keys()) and 'V_th' in list(parameter_set.net_pars.neuron_pars[pop_idx].keys()):
                    v_reset = parameter_set.net_pars.neuron_pars[pop_idx]['V_reset']
                    v_th = parameter_set.net_pars.neuron_pars[pop_idx]['V_th']
                    possible_spike_times = [t for t in idxs if (t < len(vm) - 1) and (vm[t + 1] == v_reset) and (vm[t] != v_reset)]
                    ax4.vlines(times[possible_spike_times], v_th, 50., lw=1)

                ax4.set_ylim(min(vm) - 5., 10.)
                ax4.set_ylabel(r'$\mathrm{V_{m} [mV]}$')
                ax3.set_title('Neuron {0}'.format(str(idx)))
                currents = [x for x in list(results['analog_activity'][name].keys()) if x[0] == 'I']

                if not utils.operations.empty(currents):
                    cl = ['r', 'b', 'gray']
                    for iiddxx, nn_curr in enumerate(currents):
                        ax3.plot(times, results['analog_activity'][name][nn_curr], c=cl[iiddxx], lw=1)
                    ax3.set_xlim(min(times), max(times))
                    ax3.set_ylabel(r'$\mathrm{I^{syn}} \mathrm{[nA]}$')
                else:
                    irrelevant_keys = ['single_Vm', 'single_idx']
                    other_variables = [x for x in list(results['analog_activity'][name].keys()) if x[:6] == 'single' and x
                                       not in irrelevant_keys]
                    cl = ['g', 'k', 'gray']
                    for iiddxx, nn_curr in enumerate(other_variables):
                        ax3.plot(times, results['analog_activity'][name][nn_curr], c=cl[iiddxx], lw=1)
                        ax3.set_xlim(min(times), max(times))
                        ax3.set_ylabel('{0}'.format(nn_curr))

    if display:
        pl.show(block=False)
    if save:
        assert isinstance(save, str), "Please provide filename"
        if isinstance(fig1, mpl.figure.Figure):
            # fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig1.savefig(save + results['metadata']['population_name'] + '_Figure1.pdf')
        if isinstance(fig2, mpl.figure.Figure):
            # fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig2.savefig(save + results['metadata']['population_name'] + '_Figure2.pdf')
        if isinstance(fig3, mpl.figure.Figure):
            # fig3.tight_layout()
            fig3.savefig(save + results['metadata']['population_name'] + '_Figure3.pdf')


def plot_acc(t, accs, fit_params, acc_function, title='', ax=None, display=True, save=False):
    """
    Plot autocorrelation decay and exponential fit (can be used for other purposes where an exponential fit to the
    data is suitable
    :param t:
    :param accs:
    :param fit_params:
    :param acc_function:
    :param title:
    :param ax:
    :param display:
    :param save:
    :return:
    """
    if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)):
        raise ValueError('ax must be matplotlib.axes.Axes instance.')

    if ax is None:
        fig = pl.figure()
        fig.suptitle(title)
        ax = fig.add_subplot(111)
    else:
        ax.set_title(title)

    for n in range(accs.shape[0]):
        ax.plot(t, accs[n, :], alpha=0.1, lw=0.1, color='k')

    error = np.sum((np.mean(accs, 0) - acc_function(t, *fit_params)) ** 2)
    label = r'$a = {0}, b = {1}, {2}={3}, MSE = {4}$'.format(str(np.round(fit_params[0], 2)), str(np.round(fit_params[
                                                                                                               1], 2)), r'\tau_{int}', str(np.round(fit_params[2], 2)), str(error))
    ax.errorbar(t, np.mean(accs, 0), yerr=st.sem(accs), fmt='', color='k', alpha=0.3)
    ax.plot(t, np.mean(accs, 0), '--')
    ax.plot(t, acc_function(t, *fit_params), 'r', label=label)
    ax.legend()

    ax.set_ylabel(r'Autocorrelation')
    ax.set_xlabel(r'Lag [ms]')
    ax.set_xlim(min(t), max(t))
    #ax.set_ylim(0., 1.)

    if save:
        assert isinstance(save, str), "Please provide filename"
        ax.figure.savefig(save + 'acc_fit.pdf')

    if display:
        pl.show(block=False)


def scatter_variability(variable, ax=None, display=True, save=False):
    """
    scatter the variance vs mean of a given variable
    :param variable:
    :return:
    """
    if ax is None:
        fig, ax = pl.subplots()
    else:
        fig, ax = viz.helper.check_axis(ax)
    variable = np.array(variable)
    vars = []
    means = []
    if len(np.shape(variable)) == 2:
        for n in range(np.shape(variable)[0]):
            vars.append(np.var(variable[n, :]))
            means.append(np.mean(variable[n, :]))
    else:
        for n in range(len(variable)):
            vars.append(np.var(variable[n]))
            means.append(np.mean(variable[n]))

    ax.scatter(means, vars, color='k', lw=0.5, alpha=0.3)
    x_range = np.linspace(min(means), max(means), 100)
    ax.plot(x_range, x_range, '--r', lw=2)
    ax.set_xlabel('Means')
    ax.set_ylabel('Variances')
    viz.helper.fig_output(fig, display=display, save=save)


def plot_2d_parscans(image_arrays=[], axis=[], fig_handle=None, labels=[], cmap='coolwarm', boundaries=[],
                     interpolation='nearest', display=True, **kwargs):
    """
    Plots a list of arrays as images in the corresponding axis with the corresponding colorbar

    :return:
    """
    assert len(image_arrays) == len(axis), "Number of provided arrays must match number of axes"

    origin = 'upper'
    for idx, ax in enumerate(axis):
        if not isinstance(ax, mpl.axes.Axes):
            raise ValueError('ax must be matplotlib.axes.Axes instance.')
        else:
            plt1 = ax.imshow(image_arrays[idx], aspect='auto', origin=origin, cmap=cmap, interpolation=interpolation)
            if boundaries:
                cont = ax.contour(image_arrays[idx], boundaries[idx], origin='lower', colors='k', linewidths=2)
                pl.clabel(cont, fmt='%2.1f', colors='k', fontsize=12)
            if labels:
                ax.set_title(labels[idx])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", "10%", pad="4%")
            if fig_handle is not None:
                # cbar = fig_handle.colorbar(plt1, cax=cax, format='%.2f')
                cbar = fig_handle.colorbar(plt1, cax=cax)
                cbar.ax.tick_params(labelsize=15)
            ax.set(**kwargs)
            pl.draw()
    if display:
        pl.show(block=False)


def plot_3d_volume(X):
    """

    :return:
    """
    assert has_mayavi, "mayavi required"
    b1 = np.percentile(X, 20)
    b2 = np.percentile(X, 80)
    mlab.pipeline.volume(mlab.pipeline.scalar_field(X), vmin=b1, vmax=b2)
    mlab.axes()

    arr = mlab.screenshot()
    pl.imshow(arr)


def plot_3d_parscans(image_arrays=[], axis=[], dimensions=[10, 10, 10], fig_handle=None, labels=[], cmap='jet',
                     boundaries=[],
                     **kwargs):
    """
    Plot results when 3 different parameter axes are used.. (needs further testing)
    :return:
    """
    assert has_mayavi, "mayavi required"
    assert len(image_arrays) == len(axis), "Number of provided arrays mus match number of axes"
    x = np.linspace(0, dimensions[0], 1)
    y = np.linspace(0, dimensions[1], 1)
    z = np.linspace(0, dimensions[2], 1)

    X1, Y1, Z1 = np.meshgrid(x, y, z)

    origin = 'upper'
    for idx, ax in enumerate(axis):
        if not isinstance(ax, mpl.axes.Axes):
            raise ValueError('ax must be matplotlib.axes.Axes instance.')
        else:
            plt1 = ax.imshow(image_arrays[idx], aspect='auto', interpolation='nearest', origin=origin, cmap=cmap)

            if boundaries:
                cont = ax.contour(image_arrays[idx], boundaries[idx], origin='lower', colors='k', linewidths=2)
                pl.clabel(cont, fmt='%2.1f', colors='k', fontsize=12)
            if labels:
                ax.set_title(labels[idx])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", "10%", pad="4%")
            if fig_handle is not None:
                cbar = fig_handle.colorbar(plt1, cax=cax)

            ax.set(**kwargs)
            pl.draw()
    pl.show(block=False)


def plot_w_out(w_out, label, display=True, save=False):
    """
    Creates a histogram of the readout weights
    """
    fig1, ax1 = pl.subplots()
    fig1.suptitle("{} - Biclustering readout weights".format(str(label)))
    n_clusters = np.min(w_out.shape)
    n_bars = np.max(w_out.shape)
    model = SpectralBiclustering(n_clusters=n_clusters, method='log',
                                 random_state=0)
    model.fit(w_out)
    fit_data = w_out[np.argsort(model.row_labels_)]
    fit_data = fit_data[:, np.argsort(model.column_labels_)]
    ax1.matshow(fit_data, cmap=pl.cm.Blues, aspect='auto')
    ax1.set_yticks(list(range(len(model.row_labels_))))
    ax1.set_yticklabels(np.argsort(model.row_labels_))
    ax1.set_ylabel("Out")
    ax1.set_xlabel("Neuron")

    if np.argmin(w_out.shape) == 0:
        w_out = w_out.copy().T
    ##########################################################
    fig = pl.figure()
    for n in range(n_clusters):
        ax = fig.add_subplot(1, n_clusters, n+1)
        ax.set_title(r"{} - $".format(str(label)) + r"W^{\mathrm{out}}_{" + "{}".format(n) + r"}$")
        ax.barh(list(range(n_bars)), w_out[:, n], height=1.0, linewidth=0, alpha=0.8)
        ax.set_ylim([0, w_out.shape[0]])
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    if save:
        assert isinstance(save, str), "Please provide filename"
        fig1.savefig(save+'W_out_Biclustering.pdf')
        fig.savefig(save+'w_out.pdf')
    if display:
        pl.show(block=False)


def plot_confusion_matrix(matrix, label='', ax_label=None, ax=None, display=True, save=False):
    """
    """
    if ax is not None:
        fig1, ax1 = viz.helper.check_axis(ax)
    else:
        fig1, ax1 = pl.subplots()
    if not utils.operations.empty(label):
        fig1.suptitle(r"${0}$ - Confusion Matrix".format(str(label)))
    if ax_label is not None:
        ax1.set_title(ax_label)
    ax1.matshow(matrix, cmap=pl.cm.YlGn, aspect='auto')
    viz.helper.fig_output(fig1, display, save)


def plot_raster(spike_list, dt, ax, sub_set=None, **kwargs):
    """
    Plot a nice-looking line raster
    :param spike_list: SpikeList object
    :param dt: shortest bin width
    :param ax: axis to plot on
    :param sub_set: display only subset of spiking neurons
    :param kwargs:
    :return:
    """
    if sub_set is not None:
        # plot a subset of the spiking neurons
        ids = np.random.permutation([x for ii, x in enumerate(spike_list.id_list) if spike_list.mean_rates()[ii]])[
              :sub_set]
        tmp = []
        for n_id, idd in enumerate(ids):
            tmp.append([(n_id, t) for t in spike_list.spiketrains[idd].spike_times])
        tmp = list(itertools.chain(*tmp))
        spks = signals.spikes.SpikeList(tmp, list(np.arange(sub_set)))
    else:
        spks = spike_list
    ax1a = pl.twinx(ax)
    spks.raster_plot(ax=ax, display=False, **kwargs)
    ax.grid(False)
    ax.set_ylabel(r'Neuron')
    ax.set_xlabel(r'Time $[\mathrm{ms}]$')
    ax1a.plot(spike_list.time_axis(dt)[:-1], spike_list.firing_rate(dt, average=True), 'k', lw=1., alpha=0.5)
    ax1a.plot(spike_list.time_axis(10.)[:-1], spike_list.firing_rate(10., average=True), 'r', lw=2.)
    ax1a.grid(False)
    ax1a.set_ylabel(r'Rate $[\mathrm{sps}/s]$')


def plot_target_out(target, output, time_axis=None, label='', display=False, save=False):
    """

    :param target:
    :param output:
    :param label:
    :param display:
    :param save:
    :return:
    """
    fig2, ax2 = pl.subplots()
    fig2.suptitle(label)
    if output.shape == target.shape:
        tg = target[0]
        oo = output[0]
    else:
        tg = target[0]
        oo = output[:, 0]

    if time_axis is None:
        time_axis = np.arange(tg.shape[1])

    ax2ins = zoomed_inset_axes(ax2, 0.5, loc=1)
    ax2ins.plot(tg, c='r')
    ax2ins.plot(oo, c='b')
    ax2ins.set_xlim([100, 200])
    ax2ins.set_ylim([np.min(tg), np.max(tg)])

    mark_inset(ax2, ax2ins, loc1=2, loc2=4, fc="none", ec="0.5")

    pl1 = ax2.plot(tg, c='r', label='target')
    pl2 = ax2.plot(oo, c='b', label='output')
    ax2.set_xlabel('Time [ms]')
    ax2.set_ylabel('u(t)')
    ax2.legend(loc=3)
    if display:
        pl.show(block=False)
    if save:
        pl.savefig(save + label + '_TargetOut.pdf')


def plot_connectivity_matrix(matrix, source_name, target_name, label='', ax=None,
                             display=True, save=False):
    """

    :param matrix:
    :param source_name:
    :param target_name:
    :param label:
    :param ax:
    :param display:
    :param save:
    :return:
    """
    if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)):
        raise ValueError('ax must be matplotlib.axes.Axes instance.')

    if len(label.split('_')) == 2:
        title = label.split('_')[0] + '-' + label.split('_')[1]
        label = title
    else:
        title = label
    if ax is None:
        fig, ax = pl.subplots()
        fig.suptitle(r'${0}$'.format(str(title)))
    else:
        ax.set_title(r'${0}$'.format(str(title)))

    plt1 = ax.imshow(matrix, interpolation='nearest', aspect='auto', extent=None, cmap='jet')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    pl.colorbar(plt1, cax=cax)
    ax.set_title(label)
    ax.set_xlabel('Source=' + str(source_name))
    ax.set_ylabel('Target=' + str(target_name))
    if display:
        pl.show(block=False)
    if save:
        pl.savefig(save + '{0}connectivityMatrix.pdf'.format(label))


def plot_response_activity(spike_list, input_stimulus, start=None, stop=None):
    """
    Plot population responses to stimuli (spiking activity)

    :param spike_list:
    :param input_stimulus:
    :return:
    """
    fig = pl.figure()
    ax1 = pl.subplot2grid((12, 1), (0, 0), rowspan=6, colspan=1)
    ax2 = pl.subplot2grid((12, 1), (7, 0), rowspan=2, colspan=1, sharex=ax1)

    rp = SpikePlots(spike_list, start, stop)
    plot_props = {'xlabel': 'Time [ms]', 'ylabel': 'Neuron', 'linewidth': 1.0, 'linestyle': '-'}
    rp.dot_display(ax=[ax1, ax2], with_rate=True, display=False, save=False, **plot_props)

    rp.mark_events(ax1, input_stimulus, start, stop)
    rp.mark_events(ax2, input_stimulus, start, stop)


def plot_isi_data(results, data_label, color_map='jet', location=0, fig_handles=None,
                  axes=None, display=True, save=False):
    """

    :param results:
    :param data_label:
    :param color_map:
    :param location:
    :param fig_handles:
    :param axes:
    :param display:
    :param save:
    :return:
    """
    keys = ['isi', 'cvs', 'lvs', 'lvRs', 'ents', 'iR', 'cvs_log', 'isi_5p', 'ai']
    # ISI histograms
    if fig_handles is None and axes is None:
        fig1, axes_histograms = isi_analysis_histogram_axes(label=data_label)
        fig2 = pl.figure()
    else:
        fig1 = fig_handles[0]
        fig2 = fig_handles[1]
        axes_histograms = axes
    data_list = [results[k] for k in keys]
    args = [{'xlabel': r'${0}$'.format(k), 'ylabel': r'Frequency'} for k in keys]
    bins = [100 for k in keys]
    plot_histograms(axes_histograms, data_list, bins, args, cmap=color_map)

    # ISI Summary statistics
    fig2.suptitle(data_label + 'ISI statistics')
    summary_statistics(data_list, labels=keys, loc=location, fig=fig2, cmap=color_map)

    if display:
        pl.show(block=False)
    if save:
        fig1.savefig(save + '{0}_isi_histograms.pdf'.format(str(data_label)))
        fig2.savefig(save + '{0}_isi_summary.pdf'.format(str(data_label)))


def plot_synchrony_measures(results, label='', time_resolved=False, epochs=None, color_map='jet', display=True,
                            save=False):
    """

    :param results:
    :param label:
    :param time_resolved:
    :param epochs:
    :param color_map:
    :param display:
    :param save:
    :return:
    """
    # Synchrony distance matrices
    keys = ['ISI_distance_matrix', 'SPIKE_distance_matrix', 'SPIKE_sync_matrix']
    if all([k in list(results.keys()) for k in keys]):
        fig3 = pl.figure()
        fig3.suptitle('{0} - Pairwise Distances'.format(str(label)))
        #ax31 = fig3.add_subplot(221)
        ax32 = fig3.add_subplot(222)
        ax33 = fig3.add_subplot(223)
        ax34 = fig3.add_subplot(224)
        image_arrays = [results['ISI_distance_matrix'], results['SPIKE_distance_matrix'],
                        results['SPIKE_sync_matrix']]
        plot_2d_parscans(image_arrays=image_arrays, axis=[ax32, ax33, ax34],
                         fig_handle=fig3, labels=[r'$D_{ISI}$', r'$D_{SPIKE}$', r'$D_{SPIKE_{S}}$'],
                         display=display)
        if save:
            fig3.savefig(save + '{0}_distance_matrices.pdf'.format(str(label)))
    if time_resolved:
        # Time resolved synchrony
        fig4 = pl.figure()
        fig4.suptitle('{0} - Time-resolved synchrony'.format(str(label)))
        ax1 = fig4.add_subplot(311)
        ax2 = fig4.add_subplot(312, sharex=ax1)
        ax3 = fig4.add_subplot(313, sharex=ax1)
        if epochs is not None:
            mark_epochs(ax1, epochs, color_map)

        x, y = results['SPIKE_sync_profile'].get_plottable_data()
        ax1.plot(x, y, '-g', alpha=0.4)
        ax1.set_ylabel(r'$S_{\mathrm{SPIKE_{s}}}(t)$')
        ax1.plot(x, utils.operations.smooth(y, window_len=100, window='hamming'), '-g', lw=2.5)
        ax1.set_xlim([min(x), max(x)])

        x3, y3 = results['ISI_profile'].get_plottable_data()
        ax2.plot(x3, y3, '-b', alpha=0.4)
        ax2.plot(x3, utils.operations.smooth(y3, window_len=100, window='hamming'), '-b', lw=2.5)
        ax2.set_ylabel(r'$d_{\mathrm{ISI}}(t)$')
        ax2.set_xlim([min(x3), max(x3)])

        x5, y5 = results['SPIKE_profile'].get_plottable_data()
        ax3.plot(x5, y5, '-k', alpha=0.4)
        ax3.plot(x5, utils.operations.smooth(y5, window_len=100, window='hamming'), '-k', lw=2.5)
        ax3.set_ylabel(r'$d_{\mathrm{SPIKE}}(t)$')
        ax3.set_xlim([min(x5), max(x5)])

        if save:
            fig4.savefig(save + '{0}_time_resolved_sync.pdf'.format(str(label)))

    if display:
        pl.show(block=False)


def plot_averaged_time_resolved(results, spike_list, label='', epochs=None, color_map='jet', display=True, save=False):
    """

    :param results:
    :param spike_list:
    :param label:
    :param epochs:
    :param color_map:
    :param display:
    :param save:
    :return:
    """
    # time resolved regularity
    fig5 = pl.figure()
    fig5.suptitle('{0} - Time-resolved regularity'.format(str(label)))
    stats = ['isi_5p_profile', 'cvs_profile', 'cvs_log_profile', 'lvs_profile', 'iR_profile', 'ents_profile']
    cm = get_cmap(len(stats), color_map)

    for idx, n in enumerate(stats):
        ax 			= fig5.add_subplot(len(stats), 1, idx + 1)
        data_mean 	= np.array([results[n][i][0] for i in range(len(results[n]))])
        data_std 	= np.array([results[n][i][1] for i in range(len(results[n]))])
        t_axis 		= np.linspace(spike_list.t_start, spike_list.t_stop, len(data_mean))

        ax.plot(t_axis, data_mean, c=cm(idx), lw=2.5)
        ax.fill_between(t_axis, data_mean - data_std, data_mean + data_std, facecolor=cm(idx), alpha=0.2)
        ax.set_ylabel(n)
        ax.set_xlabel('Time [ms]')
        ax.set_xlim(spike_list.time_parameters())
        if epochs is not None:
            mark_epochs(ax, epochs, color_map)

    # activity plots
    fig6 = pl.figure()
    fig6.suptitle('{0} - Activity Analysis'.format(str(label)))
    ax61 = pl.subplot2grid((25, 1), (0, 0), rowspan=20, colspan=1)
    ax62 = pl.subplot2grid((25, 1), (20, 0), rowspan=5, colspan=1)
    pretty_raster(spike_list, analysis_interval=[spike_list.t_start, spike_list.t_stop], n_total_neurons=1000, ax=ax61)
    # plot_raster(spike_list, 1., ax61, sub_set=100, **{'color': 'k', 'alpha': 0.8, 'marker': '|', 'markersize': 2})
    stats = ['ffs_profile']

    cm = get_cmap(len(stats), color_map)
    for idx, n in enumerate(stats):
        data_mean = np.array([results[n][i][0] for i in range(len(results[n]))])
        data_std = np.array([results[n][i][1] for i in range(len(results[n]))])
        t_axis = np.linspace(spike_list.t_start, spike_list.t_stop, len(data_mean))
        ax62.plot(t_axis, data_mean, c=cm(idx), lw=2.5)
        ax62.fill_between(t_axis, data_mean - data_std, data_mean +
                          data_std, facecolor=cm(idx), alpha=0.2)
        ax62.set_ylabel(r'$\mathrm{FF}$')
        ax62.set_xlabel('Time [ms]')
        ax62.set_xlim(spike_list.time_parameters())
    if epochs is not None:
        mark_epochs(ax61, epochs, color_map)
        mark_epochs(ax62, epochs, color_map)

    if display:
        pl.show(block=False)
    if save:
        fig5.savefig(save + '{0}_time_resolved_reg.pdf'.format(str(label)))
        fig6.savefig(save + '{0}_activity_analysis.pdf'.format(str(label)))


def plot_dimensionality(result, pca_obj, rotated_data=None, data_label='', display=True, save=False):
    fig7 = pl.figure()
    ax71 = fig7.add_subplot(121, projection='3d')
    ax71.grid(False)
    ax72 = fig7.add_subplot(122)

    ax71.plot(rotated_data[:, 0], rotated_data[:, 1], rotated_data[:, 2], '.-',color='r', lw=2, alpha=0.8)
    ax71.set_title(r'${0} - (3 PCs) = {1}$'.format(data_label, str(round(np.sum(
        pca_obj.explained_variance_ratio_[:3]), 1))))
    ax72.plot(pca_obj.explained_variance_ratio_, 'ob')
    ax72.plot(pca_obj.explained_variance_ratio_, '-b')
    ax72.plot(np.ones_like(pca_obj.explained_variance_ratio_) * result, np.linspace(0.,
                                                                                    np.max(pca_obj.explained_variance_ratio_), len(pca_obj.explained_variance_ratio_)),
              '--r', lw=2.5)
    ax72.set_xlabel(r'PC')
    ax72.set_ylabel(r'Variance Explained')
    ax72.set_xlim([0, round(result) * 2])
    ax72.set_ylim([0, np.max(pca_obj.explained_variance_ratio_)])
    if display:
        pl.show(block=False)
    if save:
        fig7.savefig(save + '{0}_dimensionality.pdf'.format(data_label))


def plot_synaptic_currents(I_ex, I_in, time_axis):
    fig, ax = pl.subplots()
    ax.plot(time_axis, I_ex, 'b')
    ax.plot(time_axis, I_in, 'r')
    ax.plot(time_axis, np.mean(I_ex) * np.ones_like(I_ex), 'b--')
    ax.plot(time_axis, np.mean(I_in) * np.ones_like(I_in), 'r--')
    ax.plot(time_axis, np.abs(I_ex) - np.abs(I_in), c='gray')
    ax.plot(time_axis, np.mean(np.abs(I_ex) - np.abs(I_in))*np.ones_like(I_ex), '--', c='gray')


def pretty_raster(global_spike_list, analysis_interval, sub_pop_gids=None, n_total_neurons=10, ax=None, color='k'):
    """
    Simple line raster to plot a subset of the populations (for publication)
    :return:
    """
    plot_list = global_spike_list.time_slice(t_start=analysis_interval[0], t_stop=analysis_interval[1])
    new_ids = np.intersect1d(plot_list.select_ids("cell.mean_rate() > 0"),
                             plot_list.select_ids("cell.mean_rate() < 100"))

    if ax is None:
        fig = pl.figure()
        # pl.axis('off')
        ax = fig.add_subplot(111)#, frameon=False)

    if sub_pop_gids is not None:
        assert (isinstance(sub_pop_gids, list)), "Provide a list of lists of gids"
        assert (len(sub_pop_gids) == 2), "Only 2 populations are currently covered"
        lenghts = list(map(len, sub_pop_gids))
        sample_neurons = [(n_total_neurons * n) / np.sum(lenghts) for n in lenghts]
        # id_ratios = float(min(lenghts)) / float(max(lenghts))
        # sample_neurons = [n_total_neurons * id_ratios, n_total_neurons * (1 - id_ratios)]

        neurons_1 = []
        neurons_2 = []
        while len(neurons_1) != sample_neurons[0] or len(neurons_2) != sample_neurons[1]:
            chosen = np.random.choice(new_ids, size=n_total_neurons, replace=False)
            neurons_1 = [x for x in chosen if x in sub_pop_gids[0]]
            neurons_2 = [x for x in chosen if x in sub_pop_gids[1]]
            if len(neurons_1) > sample_neurons[0]:
                neurons_1 = neurons_1[:sample_neurons[0]]
            if len(neurons_2) > sample_neurons[1]:
                neurons_2 = neurons_2[:sample_neurons[1]]

    else:
        chosen_ids = np.random.permutation(new_ids)[:n_total_neurons]
        new_list = plot_list.id_slice(chosen_ids)

        neuron_ids = [np.where(x == np.unique(new_list.raw_data()[:, 1]))[0][0] for x in new_list.raw_data()[:, 1]]
        tmp = [(neuron_ids[idx], time) for idx, time in enumerate(new_list.raw_data()[:, 0])]

    for idx, n in enumerate(tmp):
        ax.vlines(n[1] - analysis_interval[0], n[0] - 0.5, n[0] + 0.5, **{'color': color, 'lw': 1.})
    ax.set_ylim(-0.5, n_total_neurons - 0.5)
    ax.set_xlim(0., analysis_interval[1] - analysis_interval[0])
    ax.grid(False)


def scatter_projections(state, label_sequence, cmap, ax=None, display=False, save=False):
    """
    Scatter plot 3D projections from a high-dimensional state matrix
    :param state:
    :param label_sequence:
    :param cmap: color map
    :param ax:
    :param display:
    :return:
    """
    if ax is None and state.shape[1] == 3:
        fig = pl.figure()
        ax = fig.add_subplot(111, projection='3d')
    elif ax is None and state.shape[1] == 2:
        fig = pl.figure()
        ax = fig.add_subplot(111)
    else:
        fig, ax = check_axis(ax)

    unique_labels = np.unique(label_sequence)
    ccs = [np.array(cmap(i)) for i in range(len(unique_labels))]
    lab_seq = np.array(list(itertools.chain(label_sequence)))

    scatters = []
    for color, index in zip(ccs, unique_labels):
        if state.shape[1] == 3:
            tmp = ax.plot(state[np.where(lab_seq == index)[0], 0], state[np.where(lab_seq == index)[0], 1],
                             state[np.where(lab_seq == index)[0], 2], marker='o', linestyle='', ms=10, c=color,
                          alpha=0.8,
                          label=index)
        elif state.shape[1] == 2:
            tmp = ax.plot(state[np.where(lab_seq == index)[0], 0], state[np.where(lab_seq == index)[0], 1],
                          marker='o', linestyle='', ms=10, c=color, alpha=0.8, label=index)
        else:
            raise NotImplementedError("Input state matrix must be 2 or 3 dimensional")
        scatters.append(tmp[0])
    if len(unique_labels) <= 20:
        pl.legend(loc=0, handles=scatters)
    fig_output(fig, display, save)