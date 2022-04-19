import numpy as np
import itertools
import importlib

nest = None if importlib.util.find_spec("nest") is None else importlib.import_module("nest")

from fna.tools.parameters import ParameterSet
from fna.tools.network_architect.connectivity import NESTConnector
from .nest_encoder import NESTEncoder
from fna.tools import utils

logger = utils.logger.get_logger(__name__)


class InputMapper(NESTConnector):
    """
    Create input map, specifying connectivity between inputs and network
    """
    def __init__(self, source, target, parameters):
        """

        :param stimulus_set:
        :param encoding_layer:
        :param network:
        :param parameters:
        """
        # assert isinstance(source, NESTEncoder) and isinstance(target, Population), "source of input maps must be " \
        assert isinstance(source, NESTEncoder), "source of input maps must be NESTEncoder "
        self.connection_types = []
        self.connection_names = []
        self.synaptic_weights = {}
        self.synaptic_delays = {}
        self.source = source
        self.target = target
        self.total_delay = 0.

        logger.info("Connecting input generators: {0!s} -> {1!s}".format(source.name, target.name))
        if "synapse_types" not in parameters.keys():
            connection_parameters = self._parse_parameters(parameters, synapse_names=str(source.name)+'-syn')
        else:
            connection_parameters = ParameterSet(parameters)

        # store complete connection parameters
        self.parameters = connection_parameters

        # iterate over all synapse types
        for n in range(connection_parameters.n_synapse_types):
            logger.info("    - %s [%s]" % (connection_parameters.synapse_types[n], connection_parameters.syn_specs[n]['model']))

            # index of source and target population in the population lists and the gids of the elements
            src_gids = list(itertools.chain(*source.gids))
            # src_pop_idx, src_gids = self._get_population_id(connection_parameters.synapse_types[n][1], source,
            #                                                 topology=connection_parameters.topology_dependent[n])
            tget_gids = self._get_population_id(connection_parameters.synapse_types[n][0],
                                                target, topology=connection_parameters.topology_dependent[n])
            # copy and modify synapse model
            if hasattr(connection_parameters, "synapse_names"):
                synapse_name = connection_parameters.synapse_names[n]
            else:
                synapse_name = connection_parameters.synapse_types[n][1] + '_' + connection_parameters.synapse_types[n][0]

            if synapse_name not in nest.Models():
                nest.CopyModel(connection_parameters.syn_specs[n]['model'], synapse_name)
            self.connection_names.append(synapse_name)
            self.connection_types.append(connection_parameters.synapse_types[n])

            self._setup_connections(src_gids, tget_gids, synapse_name, weight_matrix=connection_parameters.weight_matrix[
                n], topology=connection_parameters.topology_dependent[n], syn_specs=connection_parameters.syn_specs[
                n], conn_specs=connection_parameters.conn_specs[n])
            source._is_connected = True

        if connection_parameters['syn_specs'][0] is not None and 'delay' in connection_parameters['syn_specs'][0].keys():
            delays = np.unique([x['delay'] for x in connection_parameters['syn_specs']])
            self.determine_total_delay(delays)
        else:
            self.determine_total_delay()

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
                src_gids = [s[0] for s in self.source.gids]  # create a clean list here, get rid of the tuples
                # tget_gids = self.target.populations[self.target.population_names.index(connection_name[0])].gids
                tget_gids = self.target.find_population(connection_name[0]).gids
                self.synaptic_weights.update({connection_name: self.extract_connection_matrix(
                    src_gids, tget_gids, key='weight', connection_name=connection_name)})
                self.synaptic_delays.update({connection_name: self.extract_connection_matrix(
                    src_gids, tget_gids, key='delay', connection_name=connection_name)})

    def determine_total_delay(self, delays=None):
        """
        Determine the connection delays involved in the encoding layer
        :return:
        """
        if delays is None:
            self.get_connections()

            for k, v in list(self.synaptic_delays.items()):
                delay = np.unique(np.array(v[v.nonzero()].todense()))
                assert (len(delay) == 1), "Heterogeneous delays in encoding layer are not supported.."
        else:
            delay = np.unique(delays)
            assert (len(delay) == 1), "Heterogeneous delays in encoding layer are not supported.."

        self.total_delay = float(delay)
        # logger.info("\t- total delays in EncodingLayer: {0} ms".format(str(self.total_delay)))
