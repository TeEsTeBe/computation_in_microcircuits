import os
import pickle as pkl

import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fna.tools import utils
from fna.tools import visualization as viz
from fna.tools.analysis import metrics
from fna.tools.utils import data_handling

logger = utils.logger.get_logger(__name__)


class StateMatrix(object):
    """
    Class to store a state/response matrix of a given population, together with some additional metadata.
    Independent of the underlying network type.
    """

    def __init__(self, matrix, label, state_var, population, sampled_times=None, standardize=False,
                 dataset_label=None, save=False):
        """

        :param matrix:
        :param label:
        :param state_var:
        :param population:
        :param sampled_times:
        :param standardize:
        :param dataset_label:
        :param save:
        """
        self.matrix = matrix
        self.state_var = state_var
        self.sampled_times = sampled_times  # stores the exact sampling times, if applicable
        self.method = None
        self.population = population
        self.label = label

        if standardize:
            self.standardize()

        if save:
            self.save(dataset_label)

    def standardize(self):
        self.matrix = StandardScaler().fit_transform(self.matrix.T).T

    def plot_sample_traces(self, n_neurons=10, time_axis=None, analysis_interval=None, display=True, save=False):
        """
        Plot a sample of the neuron responses
        :param n_neurons:
        :param time_axis:
        :param analysis_interval:
        :return:
        """
        if time_axis is None:
            time_axis = np.arange(self.matrix.shape[1])
        if analysis_interval is None:
            analysis_interval = [time_axis.min(), time_axis.max()]
        try:
            t_idx = [np.where(time_axis == analysis_interval[0])[0][0], np.where(time_axis == analysis_interval[1])[0][0]]
        except:
            raise IOError("Analysis interval bounds not found in time axis")

        fig, axes = plt.subplots(n_neurons, 1, sharex=True, figsize=(10, 2*n_neurons))
        fig.suptitle("{} [{} state] - {}".format(self.population, self.state_var, self.label))
        neuron_idx = np.random.permutation(self.matrix.shape[0])[:n_neurons]
        for idx, (neuron, ax) in enumerate(zip(neuron_idx, axes)):
            ax.plot(time_axis[t_idx[0]:t_idx[1]], self.matrix[neuron, t_idx[0]:t_idx[1]])
            ax.set_ylabel(r"$X_{"+"{0}".format(neuron)+"}$")
            ax.set_xlim(analysis_interval)
            if idx == len(neuron_idx) - 1:
                ax.set_xlabel("Time [ms]")
        viz.helper.fig_output(fig, display, save)

    def plot_matrix(self, time_axis=None, display=True, save=False):
        """
        Plot the complete state matrix
        :param time_axis:
        :return:
        """
        if time_axis is None:
            time_axis = np.arange(self.matrix.shape[1])
        fig, (ax11, ax12) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [18, 4], 'hspace': 0.1}, sharex=False)
        fig.suptitle("{} [{} states] - {}".format(self.population, self.state_var, self.label))
        _, ax11 = viz.plotting.plot_matrix(self.matrix, ax=ax11, save=False, display=False, data_label=None)
        ax11.set_ylabel('Neuron')
        ax12.plot(time_axis, self.matrix.mean(0), lw=2)
        divider2 = make_axes_locatable(ax12)
        cax2 = divider2.append_axes("right", size="5%", pad="4%")
        cax2.remove()
        ax12.set_xlabel("Time [ms]")
        ax12.set_xlim([time_axis.min(), time_axis.max()])
        ax12.set_ylabel(r"$\bar{X}$")
        viz.helper.fig_output(fig, display, save)

    def plot_trajectory(self, display=True, save=False):
        fig3 = plt.figure()
        ax31 = fig3.add_subplot(111, projection='3d')
        effective_dimensionality = metrics.analyse_state_matrix(self.matrix, stim_labels=None, epochs=None, label=None, plot=False,
                                                        display=True, save=False)['dimensionality']
        fig3.suptitle("{} [{} states] - {}".format(self.population, self.state_var, self.label) +
                      r" $\lambda_{\mathrm{eff}}="+"{}".format(effective_dimensionality)+"$")
        viz.plotting.plot_trajectory(self.matrix, label="{} [{} states] - {}".format(self.population,
                                                                                     self.state_var, self.label) +
                                           " Trajectory", ax=ax31, color='k',
                                     display=False, save=False)
        viz.helper.fig_output(fig3, display, save)

    def state_density(self):
        pass

    def save(self, dataset_label):
        """
        Save the StateMatrix object. By default only the test matrix is stored, but it's
        also possible to store the intermediate ones for each batch.

        :return:
        """
        try:
            filename = "{}_{}_{}_{}.pkl".format(data_handling.filename_prefixes['state_matrix'], self.label,
                                                data_handling.data_label, dataset_label)
            with open(os.path.join(data_handling.paths['activity'], filename), 'wb') as f:
                pkl.dump(self, f)
        except Exception as e:
            logger.warning("Could not save StateMatrix {}, storage paths not set?".format(self.label))