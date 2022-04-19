import time
import os
import pickle as pkl
import importlib
import numpy as np
from scipy.sparse import lil_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

nest = None if importlib.util.find_spec("nest") is None else importlib.import_module("nest")

from fna.tools import parameters
from fna.tools import utils
from fna.tools import network_architect as net
from fna.tools import visualization as viz
from fna.tools.utils.operations import iterate_obj_list, copy_dict, empty
from fna.tools.utils import data_handling

logger = utils.logger.get_logger(__name__)


def verify_consistent_dimensions(w, src_gids, tget_gids):
    """
    Verify whether the connecsstion matrix complies with the (source and target) population sizes
    """
    src_dim, tget_dim = w.shape
    if len(src_gids) == src_dim and len(tget_gids) == tget_dim:
        return True
    else:
        return False


# ######################################################################################################################
class ConnectionMapper(ABC):
    """
    ConnectionMapper is a wrapper for connectivity structures, in all forms used in this project and connecting any
    systems: input to network, recurrent connections, internal network connections, output connections (...)
    Each ConnectionMapper object holds all the relevant properties of all connections among two networks or populations
    """
    @abstractmethod
    def set_connections(self):
        pass

    @abstractmethod
    def get_connections(self):
        pass

    def plot_distributions(self):
        pass

    def plot_spectral_radius(self):
        pass

    def plot_weights(self):
        pass

    def plot_connections(self):
        pass

    def save(self, data_label):
        try:
            filename = "{}_{}_{}.pkl".format(data_handling.filename_prefixes['connectivity'],
                                             data_handling.data_label, data_label)
            with open(os.path.join(data_handling.paths['system'], filename), 'wb') as f:
                pkl.dump(self, f)
        except Exception as e:
            logger.warning("Could not save ConnectionMapper {}, storage paths not set?".format(data_label))


class TFConnector(ConnectionMapper):
    """
    Specific connectivity implementations for TensorFlow graphs
    """

    def __init__(self, network, connection_parameters):
        """
        TFConnector constructor
        :param network: [ContinuousRateRNN or ArtificialNeuralNetwork object]
        :param connection_parameters: [dict] instructions for connecting the populations
        """
        self.connection_types = []
        self.connection_names = []
        self.synaptic_weights = {k: [] for k in connection_parameters.keys()}

        logger.info("Connecting network: {0!s}".format(network.name))

        for k, v in connection_parameters.items():
            logger.info("    - %s" % k)
            if isinstance(v, dict):
                w = self.generate_weights(dimensions=v['dimensions'], density=v['density'],
                                       gen_fcn=v['distribution'][0], fcn_params=v['distribution'][1])
            elif isinstance(v, np.ndarray):
                w = v
            else:
                raise NotImplementedError("Weight matrices as {} cannot be used".format(str(type(v))))
            self.set_connections(connection_name=k, weights=w)
            self.connection_types.append(k.split('_')[1])
            self.connection_names.append(k)

        # store complete connection parameters
        if not isinstance(connection_parameters, parameters.ParameterSet):
            connection_parameters = parameters.ParameterSet(connection_parameters)
        self.parameters = connection_parameters

    def set_connections(self, connection_name=None, weights=None):
        """
        Specify the network weights
        :return:
        """
        self.synaptic_weights.update({connection_name: weights})

    @staticmethod
    def generate_weights(dimensions, density, gen_fcn, fcn_params):
        """
        Generate weight matrix
        :param dimensions: [tuple] (n_src x n_tget)
        :param density: [float] connection density
        :param gen_fcn: [np.random function] function to draw the weights from
        :param fcn_params: [dict] function parameters
        """
        w = gen_fcn(size=dimensions, **fcn_params)
        w *= (np.random.rand(*dimensions) < density)
        return np.float32(w)

    def get_connections(self):
        return self.synaptic_weights

    def plot_distributions(self):
        pass

    def plot_spectral_radius(self):
        pass

    def plot_weights(self):
        pass

    def plot_connections(self):
        pass

    # def save(self):
    #     pass


# ######################################################################################################################
class NESTConnector(ConnectionMapper):
    """
    Specific connectivity implementations for the NEST simulator
    """

    @staticmethod
    def _parse_parameters(syn_pars, synapse_names=None, topology=None):
        """
        Parse a simplified parameter set and expand it
        :param syn_pars: [dict or ParameterSet] - should have the following structure
        {'connect_populations': list of (target_label, source_label) tuples,
         'weight_matrix': list of arrays for each connection (),
         'neurons': list of neuron parameter sets or dictionaries}
        :return:
        """
        if isinstance(syn_pars, dict):
            syn_pars = parameters.ParameterSet(syn_pars)
        synapses = syn_pars.connect_populations

        # check if synapse_names is defined in the parameters
        if synapse_names is None and hasattr(syn_pars, 'synapse_names'):
            synapse_names = syn_pars.synapse_names

        if synapse_names is None:
            synapse_names = [n[1] + n[0] for n in synapses]
        elif isinstance(synapse_names, str):
            synapse_names = [synapse_names for _ in synapses]
        if topology is None:
            topology = [False for _ in range(len(synapses))]
        elif not (isinstance(topology, list) and isinstance(topology[0], dict)):
            topology = [topology for _ in range(len(synapses))]
        assert (
            np.mean([n in synapses for n in syn_pars.connect_populations]).astype(bool)), "Inconsistent Parameters"
        connection_pars = {
            'n_synapse_types': len(synapses),
            'synapse_types': synapses,
            'synapse_names': synapse_names,
            'topology_dependent': topology,
            'weight_matrix': syn_pars.weight_matrix,
            'conn_specs': syn_pars.conn_specs,
            'syn_specs': syn_pars.syn_specs}
        return parameters.ParameterSet(connection_pars)

    @staticmethod
    def _get_population_id(population_name, network, topology=False):
        """
        Get the id of the population
        :param population_name: [str]
        :param network: Network object
        :return:
        """
        pop = network.find_population(population_name)

        if topology:
            return pop.layer_gid
        else:
            return pop.gids

    @staticmethod
    def _setup_connections(src_gids, tget_gids, synapse_name, weight_matrix=None, topology=False,
                          syn_specs=None, conn_specs=None):
        """
        Setup connections among populations
        :param src_gids: [list or tuple]
        :param tget_gids: [list or tuple]
        :param synapse_name: [str]
        :param weight_matrix: [array, str with path to stored array or None]
        :param topology: [dict or bool]
        :param syn_specs: [dict or None]
        :param conn_specs: [dict or None]
        """
        # 1) if pre-computed weight matrices are given
        if (weight_matrix is not None) and (not topology):
            if isinstance(weight_matrix, str):
                w = np.load(weight_matrix)
            else:
                w = weight_matrix

            if verify_consistent_dimensions(w, src_gids, tget_gids):
                for preSyn_matidx, preSyn_gid in tqdm(enumerate(src_gids), desc="Connecting"):
                    postSyn_matidx = w[preSyn_matidx, :].nonzero()[0]
                    postSyn_gids = list(postSyn_matidx + min(tget_gids))
                    weights = [w[preSyn_matidx, x] for x in postSyn_matidx]

                    if not isinstance(syn_specs['delay'], dict) and isinstance(
                            syn_specs['delay'], list):
                        delays = syn_specs['delay']
                    elif isinstance(syn_specs['delay'], dict):
                        delays = [copy_dict(syn_specs['delay']) for _ in range(len(weights))]
                    else:
                        delays = np.repeat(syn_specs['delay'], len(weights))

                    update_dict = {
                        'model': synapse_name,
                        'weight': np.reshape(weights, (len(weights), 1)),
                        'delay': np.reshape(delays, (len(delays), 1))}

                    syn_dict = copy_dict(syn_specs, update_dict)
                    nest.Connect([preSyn_gid], postSyn_gids, syn_spec=syn_dict)
            else:
                raise Exception("Dimensions of W are inconsistent with population sizes")

        # 2) if no pre-computed weights, and no topology in pre/post-synaptic populations, use dictionaries
        elif (weight_matrix is None) and (not topology):
            syn_dict = copy_dict(syn_specs, {'model': synapse_name})
            nest.Connect(src_gids, tget_gids, conn_spec=conn_specs, syn_spec=syn_dict)

        # 3) if no precomputed weights, but topology in populations and topological connections
        elif topology:
            from nest import topology as tp
            # assert nest.is_sequence_of_gids(src_gids) and len(src_gids) == 1, "Source ids are not topology layer"
            # assert nest.is_sequence_of_gids(tget_gids) and len(tget_gids) == 1, "Target ids are not topology layer"
            tp_dict = copy_dict(topology, {'synapse_model': synapse_name})
            tp.ConnectLayers(src_gids, tget_gids, tp_dict)

    def __init__(self, source_network, target_network, connection_parameters, topology=None):
        """
        NESTConnector constructor
        :param source_network: [Network or Population object]
        :param target_network: [Network or Population object]
        :param connection_parameters: [dict] instructions for connecting the populations
        :param topology: list of dictionaries with topology connection parameters or None
        """
        assert isinstance(source_network, net.Network) or isinstance(source_network, net.Population), \
            "source and target must be Network or Population objects"
        self.connection_types = []
        self.connection_names = []
        self.synaptic_weights = {}
        self.synaptic_delays = {}
        self.source = source_network
        self.target = target_network

        logger.info("Connecting networks: {0!s} -> {1!s}".format(source_network.name, target_network.name))
        if "synapse_types" not in connection_parameters.keys():
            connection_parameters = self._parse_parameters(connection_parameters, topology=topology)
        # store complete connection parameters
        self.parameters = connection_parameters

        # iterate over all synapse types
        for n in range(connection_parameters.n_synapse_types):
            logger.info("    - %s [%s]" % (connection_parameters.synapse_types[n],
                                           connection_parameters.syn_specs[n]['model']))

            # index of source and target population in the population lists and the gids of the elements
            src_gids = self._get_population_id(connection_parameters.synapse_types[n][1], source_network,
                                               topology=connection_parameters.topology_dependent[n])
            tget_gids = self._get_population_id(connection_parameters.synapse_types[n][0],
                                                target_network, topology=connection_parameters.topology_dependent[n])
            # copy and modify synapse model
            if hasattr(connection_parameters, "synapse_names"):
                synapse_name = connection_parameters.synapse_names[n]
            else:
                synapse_name = connection_parameters.synapse_types[n][1] + '_' + connection_parameters.synapse_types[n][0]

            nest.CopyModel(connection_parameters.syn_specs[n]['model'], synapse_name)
            self.connection_names.append(synapse_name)
            self.connection_types.append(connection_parameters.synapse_types[n])

            self._setup_connections(src_gids, tget_gids, synapse_name, weight_matrix=connection_parameters.weight_matrix[
                n], topology=connection_parameters.topology_dependent[n], syn_specs=connection_parameters.syn_specs[
                n], conn_specs=connection_parameters.conn_specs[n])
            source_network._is_connected = True
            target_network._is_connected = True

    @staticmethod
    def extract_connection_matrix(src_gids, tgets_gids, key='weight', connection_name=None, progress=False):
        """
        Extract the synaptic property (weights or delays) matrix referring to connections from src_gids to tgets_gids.
        :param src_gids: list or tuple of gids of source neurons
        :param tgets_gids: list or tuple of gids of target neurons
        :param key: [str] 'weight' or 'delay'
        :param connection_name: [str] name of connection (for logging)
        :param progress: display progress bar
        :return: len(src_gids) x len(tgets_gids) weight matrix
        """
        assert key == 'weight' or key == 'delay', 'Error! Key must be weight or delay!'
        if progress:
            logger.info("Extracting {0!s} {1!s} matrix...".format(connection_name, key))
        t_start = time.time()
        matrix = lil_matrix((len(tgets_gids), len(src_gids)))
        a = nest.GetConnections(list(np.unique(src_gids)), list(np.unique(tgets_gids)))
        min_tgets_gid = min(tgets_gids)
        min_src_gid = min(src_gids)

        iterations = 100
        its = np.arange(0, len(a) + iterations, iterations).astype(int)
        its[-1] = len(a)
        # for nnn, it in tqdm(enumerate(its), desc="iterating connections", total=len(its)):
        for nnn, it in enumerate(its):
            if nnn < len(its) - 1:
                connections = a[it:its[nnn + 1]]
                st = nest.GetStatus(connections, keys=key)
                for idx, conn in enumerate(connections):
                    matrix[conn[1] - min_tgets_gid, conn[0] - min_src_gid] += st[idx]
        t_stop = time.time()
        if progress:
            logger.info("Elapsed time: {0!s} s".format(str(t_stop - t_start)))
        # for consistency with weight_matrix, we transpose this matrix (should be [src X tget])
        return matrix.T

    def get_connections(self, src_gids=None, tget_gids=None, connection_name=None):
        """
        Read and store the weights for all the connected populations, or just for
        the provided sources and targets.
        :param src_gids: list of gids of source neurons
        :param tget_gids: list of gids of target neurons
        :param connection_name: string with name for this connection (for logging)
        """
        if src_gids is not None and tget_gids is not None:
            syn_name = str(nest.GetStatus(nest.GetConnections([src_gids[0]], [tget_gids[0]]))[0]['synapse_model'])
            self.synaptic_weights.update({syn_name: self.extract_connection_matrix(src_gids, tget_gids, key='weight',
                                                                              connection_name=connection_name)})
            self.synaptic_delays.update({syn_name: self.extract_connection_matrix(src_gids, tget_gids, key='delay',
                                                                             connection_name=connection_name)})
        else:
            for connection_name in self.connection_types:
                logger.info("Extracting {} weights and delays".format(connection_name))
                # src_idx = self.source.population_names.index(connection_name[1])
                # tget_idx = self.target.population_names.index(connection_name[0])
                src_gids = self.source.populations[connection_name[1]].gids
                tget_gids = self.target.populations[connection_name[0]].gids
                self.synaptic_weights.update({connection_name: self.extract_connection_matrix(
                    src_gids, tget_gids, key='weight', connection_name=connection_name)})
                self.synaptic_delays.update({connection_name: self.extract_connection_matrix(
                    src_gids, tget_gids, key='delay', connection_name=connection_name)})

    def set_connections(self, connection_name=None, weights=None, delays=None):
        """
        Specify the network weights
        :return:
        """
        self.synaptic_weights.update({connection_name: weights})
        self.synaptic_delays.update({connection_name: delays})

    def compile_weights(self):
        """
        Gather all connection weights and delays for all connections between source and target networks
        :return full_weights: complete matrix of synaptic weights
        """
        if empty(self.synaptic_weights):
            self.get_connections()
        n_pre = sum(list(iterate_obj_list(self.source.n_neurons)))
        n_post = sum(list(iterate_obj_list(self.target.n_neurons)))
        source_population_sizes = [n.size for n in self.source.populations.values()]
        target_population_sizes = [n.size for n in self.target.populations.values()]

        full_weights = np.zeros((n_pre, n_post))

        for connection, weights in list(self.synaptic_weights.items()):
            src_idx = self.source.population_names.index(connection[1])
            tget_idx = self.target.population_names.index(connection[0])
            if src_idx == 0:
                srcs_index = [int(0), int(source_population_sizes[src_idx])]
            else:
                srcs_index = [int(np.sum(source_population_sizes[:src_idx])), int(np.sum(source_population_sizes[:src_idx+1]))]
            if tget_idx == 0:
                tgets_index = [int(0), int(target_population_sizes[tget_idx])]
            else:
                tgets_index = [int(np.sum(target_population_sizes[:tget_idx])), int(np.sum(target_population_sizes[:tget_idx+1]))]

            full_weights[srcs_index[0]:srcs_index[1], tgets_index[0]:tgets_index[1]] = np.array(weights.todense())
        return full_weights

    def plot_weights(self, save=False, display=True, single=False):
        """
        Plot weight matrices
        """
        if empty(self.synaptic_weights):
            self.get_connections()

        dims = [n.shape for n in self.synaptic_weights.values()]
        (n_rows, n_cols) = (np.sum([x[0] for x in dims]), np.sum(x[1] for x in dims))
        plot_grid = (n_rows, n_cols)

        if not single:
            fig = plt.figure()
            x_loc = 0
            y_loc = 0
            for idx, (k, v) in enumerate(self.synaptic_weights.items()):
                ax = plt.subplot2grid(plot_grid, loc=(x_loc, y_loc), rowspan=v.shape[0], colspan=v.shape[1], fig=fig)
                if not idx % 2:
                    x_loc = v.shape[0]
                else:
                    y_loc = v.shape[1]
                ax.set_title("{0} - density={1}".format(k, v.size/np.prod(v.shape)))
                viz.plotting.plot_matrix(v.todense(), labels=None, ax=ax, save=save, display=False, data_label=None)
        else:
            if len(self.synaptic_weights) > 1:
                full_weights = self.compile_weights()
            else:
                full_weights = [v for v in self.synaptic_weights.values()][0]
            fig, ax = plt.subplots()
            fig.suptitle("Global connectivity - density={0}".format(len(full_weights.nonzero()[0])/full_weights.size))
            viz.plotting.plot_matrix(full_weights, labels=None, ax=ax, save=save, display=display, data_label=None)

    def save(self, data_label):
        if empty(self.synaptic_weights):
            self.get_connections()
        try:
            filename = "{}_{}_{}.pkl".format(data_handling.filename_prefixes['connectivity'],
                                             data_handling.data_label, data_label)
            with open(os.path.join(data_handling.paths['system'], filename), 'wb') as f:
                pkl.dump(self, f)
        except Exception as e:
            logger.warning("Could not save ConnectionMapper {}, storage paths not set?".format(data_label))


def spectral_radius(w):
    """
    Compute the spectral radius of a matrix
    :param w: input matrix
    :return: spectral radius
    """
    return np.linalg.eigvals(w)


def compute_density(A):
    return np.count_nonzero(A) / float(A.shape[0] * A.shape[1])


# def print_weight_matrix(W, label=None, ax=None, cmap='Greys', save=False):
#     """
#     E/D
#     This is version 1 as per the documentation. In this case we have no real control over p_u
#     at population level, rather only at single neuron level within a stimulus-specific sub-population.
#     :return:
#     """
#     if ax is None:
#         fig = pl.figure()
#         ax = fig.add_subplot(111)
#
#     plot = ax.imshow(W, cmap=cmap, interpolation="none", aspect='auto')
#
#     if cmap != 'Greys':
#         # cax = divider.append_axes("right", "5%", pad="3%")
#         cbar = pl.colorbar(plot)
#
#     ax.grid(False)
#     ax.tick_params(labelsize=24)
#
#     # for label in ax.xaxis.get_ticklabels()[::2]:
#     # label.set_visible(False)
#
#     if save:
#         fig.tight_layout()
#         fig.savefig('{}.pdf'.format(label))
#
#
# def plot_connection_matrices(W, stim_segments, n_stim, title, save):
#     """
#
#     :param W:
#     :param stim_segments:
#     :param n_stim:
#     :param save:
#     :return:
#     """
#     fig = pl.figure(figsize=(6, 6))
#     ax = pl.subplot2grid((1, 1), (0, 0))
#     print_weight_matrix(W, ax=ax)
#
#     try:
#         seg_colors = sns.color_palette("Set2", n_stim)
#     except:
#         seg_colors = ['g', 'r', 'b', 'm', 'y']
#
#     for seg_idx, seg in enumerate(stim_segments):
#         src_start = seg[0][0]
#         src_end = seg[0][1]
#         tgt_start = seg[1][0]
#         tgt_end = seg[1][1]
#         rect = patches.Rectangle((tgt_start, src_start), tgt_end - tgt_start, src_end - src_start,
#                                  linewidth=1.5, edgecolor=seg_colors[seg_idx], facecolor='none')
#         ax.add_patch(rect)
#
#     if title:
#         ax.set_title(title, fontsize=10)
#
#     ax.tick_params(axis='both', which='major', labelsize=10)
#
#     fig.tight_layout()
#     fig.savefig(save)


