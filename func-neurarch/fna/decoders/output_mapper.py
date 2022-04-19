import os
import pickle as pkl

import numpy as np

from fna import tools
from fna.tools import visualization as viz
from fna.tools.network_architect.connectivity import ConnectionMapper
from fna.tools.utils import data_handling as io
from fna.tools.utils.operations import empty

logger = tools.utils.logger.get_logger(__name__)


class OutputMapper(ConnectionMapper):
    """
    Container for the readout weights, independent of the underlying system. Exposes useful analysis functions.
    """
    def __init__(self, label, readout_parameters):
        self.label = label
        self.extractor = readout_parameters.extractor
        self.algorithm = readout_parameters.algorithm
        self.task = readout_parameters.task

        self.batch_labels = []
        self.batch_weights = []
        self.weights = None  # the final set of weights

    def set_connections(self):
        pass

    def get_connections(self):
        pass

    def update_weights(self, batch_label, new_weights, save=False):
        """
        Copy the trained readout weights for the given batch.

        :param batch_label: [str] batch label
        :param new_weights: [np.array] weights from the latest training batch
        :param save: [bool] if True, the intermediate weights are also stored for each batch
        :return:
        """
        logger.info("Updating OutputMapper with data from batch {}".format(batch_label))
        if save:
            self.batch_labels.append(batch_label)
            self.batch_weights.append(new_weights)
        self.weights = new_weights

    def plot_distributions(self, plot_every=100):
        # if not empty(self.weights.batch_weights):
        # plot_grid = int(np.ceil(np.sqrt(len(self.weights.batch_weights))))
        #     fig, ax = plt.subplots(plot_grid, plot_grid)
        # axes = list(itertools.chain(*ax))
        # viz.helper.plot_histograms(axes[:len(self.weights.batch_weights)], self.batch_weights)
        pass

    def plot_spectral_radius(self):
        logger.warning("Spectral radius can only be computed on square matrices")
        pass

    def plot_weights(self, display=True, save=False):
        """
        Plots a histogram with the current weights.
        """
        data_label = self.label + '-' + self.algorithm + '-' + self.task
        if len(self.weights.shape) == 1:
            w = self.weights.reshape(-1, 1).T
        else:
            w = self.weights
        tools.visualization.plotting.plot_w_out(w, label=data_label, display=display,
                                                save=save)

    def measure_stability(self):
        """
        Determine the stability of the solution (norm of weights)
        """
        if not empty(self.batch_weights):
            batch_norm = [np.linalg.norm(x) for x in self.batch_weights]
        else:
            batch_norm = []
        return np.linalg.norm(self.weights), batch_norm

    def plot_connections(self):
        pass

    def save(self, label=None):
        """
        Save the OutputMapper object. By default only the last set of weights are stored, but it's
        also possible to store the intermediate ones for each batch.

        :return:
        """
        try:
            if label is None:
                filename = "{}_{}_{}_{}_{}.pkl".format(io.filename_prefixes['output_mapper'], io.data_label, self.label,
                                                       self.algorithm, self.task)
            else:
                filename = "{}_{}_{}_{}_{}.pkl".format(io.filename_prefixes['output_mapper'], label, self.label,
                                                       self.algorithm, self.task)
            with open(os.path.join(io.paths['results'], filename), 'wb') as f:
                pkl.dump(self, f)
        except Exception as e:
            logger.warning("Could not save OutputMapper {}, storage paths not set?".format(self.label))
