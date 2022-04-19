import copy
import itertools
import time
import warnings
import numpy as np

import nest

from fna.networks.snn.analysis import single_neuron_dcresponse, single_neuron_ratetransfer
from fna import tools
from fna.tools import parameters, signals
from fna.tools.network_architect import Network, Population, merge_subpopulations
from fna.tools.network_architect.topology import setup_network_topology
from fna.tools.parameters import ParameterSet
from fna.tools.utils.operations import iterate_obj_list, copy_dict, empty
from fna.tools.visualization.helper import progress_bar
from fna.decoders.extractors import SpikingExtractor
from fna.decoders import Decoder
from fna.tasks import preprocessing

logger = tools.utils.logger.get_logger(__name__)


class SpikingNetwork(Network):
    """
    Wrapper for SNNs, inheriting the properties of generic Network object and extending
    them with specific methods
    """

    def __init__(self, network_parameters, label=None, topologies=None, spike_recorders=None, analog_recorders=None):
        """
        SNN instance constructor.
        Accepts 2 types of parameter sets as input, the complete network parameters (see Network class) or a more
        compact and simpler ParameterSet with the structure described below.
        By default, devices are connected and initial states randomized, to avoid redundant calls. So, make sure all
        the relevant information is provided in the network_parameters
        :param network_parameters: [dict or ParameterSet] - should have the following structure
        {'populations': list of strings, 'population_size': list of integers, 'neurons': list of neuron parameter
        sets or dictionaries}
        :param label: [str] name tag for the architecture
        :param topologies: [list of dictionaries or None] list of population topology dictionaries
        :param spike_recorders: [list of dictionaries or None]
        :param analog_recorders: [list of dictionaries or None]
        """
        self.name = label
        self.simulator = 'NEST'
        logger.info("Initializing {0!s} architecture ({1!s}-simulated)".format(self.name, self.simulator))
        if "pop_names" not in network_parameters.keys():
            network_parameters = self._parse_parameters(network_parameters, topology=topologies,
                                                        spike_recorders=spike_recorders,
                                                        analog_recorders=analog_recorders)
        self._parameters = network_parameters

        self.populations, self.merged_populations = self._create_populations(network_parameters)
        self.n_populations = network_parameters.n_populations
        self.n_neurons = network_parameters.n_neurons
        self.population_names = network_parameters.pop_names
        self._record_spikes = network_parameters.record_spikes
        self._record_analogs = network_parameters.record_analogs
        self._spike_device_pars = network_parameters.spike_device_pars
        self._analog_device_pars = network_parameters.analog_device_pars
        self._is_connected = False
        self.device_names = [[] for _ in range(self.n_populations)]
        self.device_gids = [[] for _ in range(self.n_populations)]
        self.n_devices = [[] for _ in range(self.n_populations)]
        self.device_type = [[] for _ in range(self.n_populations)]
        self._devices_connected = False

        # # dictionary with population state matrices {pop_name: {state_matrix_label: matrix} }
        # self.state_matrices = dict()
        self.spiking_activity = [[] for _ in range(self.n_populations)]
        self.analog_activity = [[] for _ in range(self.n_populations)]

        # training
        self.batch_cnt = 0
        self.prev_batch_data = {}

        self.stim_onset_time = 0.1
        self.state_extractor = None
        self.decoders = None

        self.connect_recording_devices()
        self.initialize_states()
        self.report()

    def _create_populations(self, net_pars):
        """
        Creates individual populations and subsequently merges them if specified.
        :param net_pars: [dict] network parameters
        :return: [dict, dict] dictionaries with single and merged populations: {name: Population object}
        """
        def __create_single_population(pop_idx):
            pop_dict = {k: v[pop_idx] for k, v in net_pars.items() if k not in not_allowed_keys}
            neuron_dict = net_pars.neuron_pars[pop_idx]
            pop_name = net_pars.pop_names[pop_idx]
            if not isinstance(pop_name, str):
                raise TypeError("Population name ({}) must be of type string!".format(pop_name))

            # create neuron model named after the population
            nest.CopyModel(neuron_dict['model'], pop_name)
            # set default parameters
            nest.SetDefaults(pop_name, parameters.extract_nestvalid_dict(neuron_dict, param_type='neuron'))
            if net_pars.topology[pop_idx]:
                populations[pop_name], gids = setup_network_topology(population_dictionary=pop_dict,
                                                                     population_name=pop_name, is_subpop=False)
            else:
                # create population
                gids = nest.Create(pop_name, n=int(net_pars.n_neurons[pop_idx]))
                gids = sorted(gids)
                # set up population objects
                pop_dict.update({'gids': gids, 'is_subpop': False})
                populations[pop_name] = Population(parameters.ParameterSet(pop_dict))
            logger.info("- Population {0!s}, with ids [{1!s}-{2!s}]".format(pop_name, min(gids), max(gids)))

        # specify the keys not to be passed to the population objects
        not_allowed_keys = ['_url', 'label', 'n_populations', 'parameters',
                            'names', 'description', '', 'integration', 'merged_populations']

        populations = {}
        # iterate through the population list
        logger.info("Creating populations:")
        for n in range(net_pars.n_populations):
            __create_single_population(n)

        merged_populations = {}
        # check for and create merged populations if needed
        if 'merged_populations' in net_pars:
            for m_pop_names in net_pars.merged_populations:
                m_pop_name, m_pop = self.get_merged_population(m_pop_names, populations=populations, store=False)
                merged_populations[m_pop_name] = m_pop

        return populations, merged_populations

    def initialize_states(self, randomization_parameters=None):
        """
        Initialize the network state variables according to the distributions specified. If this information was
        provided in the network parameters, they will be read. Otherwise, provide a dictionary with the following
        structure {variable_name: (randomization_function, {function_parameters})}, e.g.:
        {'V_m': (np.random.uniform, {'low': -70., 'high': -50.})

        :param randomization_parameters: explicit randomization parameters
        """
        logger.info("Initializing state variables:")
        if randomization_parameters is None and "randomize" in self._parameters.keys():
            randomization_parameters = self._parameters.randomize
        # TODO for now, this only works when no merged populations are defined in the network parameter set
        for idx, pop_name in enumerate(self._parameters.pop_names):
            randomization_parameters = list(iterate_obj_list(randomization_parameters))
            randomize = randomization_parameters[idx]
            for k, v in list(randomize.items()):
                self.populations[pop_name].initialize_states(k, randomization_function=v[0], **v[1])

    def clone(self, original_parameter_set, devices=True, decoders=True, display=True):
        """
        Creates a new network_architect object.

        :param original_parameter_set:
        :param devices:
        :param decoders:
        :param display:
        :return:
        """
        param_set = copy.deepcopy(original_parameter_set)

        def create_clone(net):
            """
            Create Network object
            :return:
            """
            neuron_pars = [0 for _ in range(len(list(iterate_obj_list(net.populations))))]
            topology = [0 for _ in range(len(list(iterate_obj_list(net.populations))))]
            topology_dict = [0 for _ in range(len(list(iterate_obj_list(net.populations))))]
            pop_names = [0 for _ in range(len(list(iterate_obj_list(net.populations))))]
            spike_device_pars = [0 for _ in range(len(list(iterate_obj_list(net.populations))))]
            analog_dev_pars = [0 for _ in range(len(list(iterate_obj_list(net.populations))))]
            status_elements = ['archiver_length', 'element_type', 'frozen', 'global_id',
                               'has_connections', 'local', 'recordables', 't_spike',
                               'thread', 'thread_local_id', 'vp', 'n_synapses',
                               'local_id', 'model', 'parent']
            for idx_pop, pop_obj in enumerate(list(iterate_obj_list(net.populations))):
                # get generic neuron_pars (update individually later):
                neuron = parameters.extract_nestvalid_dict(nest.GetStatus([pop_obj.gids[0]])[0], param_type='neuron')
                d_tmp = {k: v for k, v in list(neuron.items()) if k not in status_elements}
                neuron_pars[idx_pop] = copy_dict(d_tmp, {'model': nest.GetStatus([pop_obj.gids[0]])[0][
                    'model']})
                if isinstance(pop_obj.topology, dict):
                    topology[idx_pop] = True
                    topology_dict[idx_pop] = pop_obj.topology
                else:
                    topology[idx_pop] = False
                    topology_dict[idx_pop] = pop_obj.topology
                pop_names[idx_pop] = net.population_names[idx_pop] + '_clone'
                if net.record_spikes[idx_pop]:
                    spike_device_pars[idx_pop] = copy_dict(net.spike_device_pars[idx_pop],
                                                                                  {'label': net.spike_device_pars[idx_pop][
                                                                                    'label'] + '_clone'})
                else:
                    spike_device_pars[idx_pop] = None
                if net.record_analogs[idx_pop]:
                    analog_dev_pars[idx_pop] = copy_dict(net.analog_device_pars[idx_pop],
                                                                                {'label': net.analog_device_pars[idx_pop][
                                                                                  'label'] + '_clone'})
                else:
                    analog_dev_pars[idx_pop] = None

            network_parameters = {'n_populations': net.n_populations,
                                  'pop_names': pop_names,
                                  'n_neurons': net.n_neurons,
                                  'neuron_pars': neuron_pars,
                                  'topology': topology,
                                  'topology_dict': topology_dict,
                                  'record_spikes': net.record_spikes,
                                  'spike_device_pars': spike_device_pars,
                                  'record_analogs': net.record_analogs,
                                  'analog_device_pars': analog_dev_pars}
            clone_net = Network(parameters.ParameterSet(network_parameters, label='clone'))
            # clone_net.connect_devices()

            for pop_idx, pop_obj in enumerate(clone_net.populations):
                for n_neuron in range(net.n_neurons[pop_idx]):
                    src_gid = net.populations[pop_idx].gids[n_neuron]
                    tget_gid = pop_obj.gids[n_neuron]
                    status_dict = nest.GetStatus([src_gid])[0]
                    st = {k: v for k, v in list(status_dict.items()) if k not in status_elements}
                    nest.SetStatus([tget_gid], st)
            return clone_net

        def connect_clone(network, clone, progress=False):
            """
            Connect the populations in the clone network_architect
            (requires iteration through all the connections in the mother network_architect...)
            :param network:
            :param clone:
            :return:
            """
            devices = ['stimulator', 'structure']
            base_idx = min(list(itertools.chain(*[n.gids for n in clone.populations]))) - 1
            logger.info("\n Replicating connectivity in clone network_architect (*)")
            for syn_idx, synapse in enumerate(network.connection_names):
                start = time.time()
                copy_synapse_name = synapse + '_clone'
                nest.CopyModel(synapse, copy_synapse_name)
                logger.info("\t- {0}".format(str(network.connection_types[syn_idx])))
                conns = nest.GetConnections(synapse_model=synapse)
                # ##
                iterate_steps = 100
                its = np.arange(0, len(conns) + iterate_steps, iterate_steps).astype(int)
                its[-1] = len(conns)
                # ##
                clone.connection_names.append(copy_synapse_name)
                conn_types = network.connection_types[syn_idx]
                connection_type = (conn_types[0] + '_clone', conn_types[1] + '_clone')
                clone.connection_types.append(connection_type)

                # src_idx = clone.population_names.index(connection_type[1])
                # tget_idx = clone.population_names.index(connection_type[0])
                # src_gids = clone.populations[src_idx].gids
                # tget_gids = clone.populations[tget_idx].gids
                # base_idx = min(list(itertools.chain(*[src_gids, tget_gids]))) - 1

                for nnn, it in enumerate(its):
                    if nnn < len(its) - 1:
                        con = conns[it:its[nnn + 1]]
                        st = nest.GetStatus(con)
                        source_gids = [x['source'] + base_idx for x in st if nest.GetDefaults(nest.GetStatus([x[
                                                                                                                  'target']])[
                                                                                                  0]['model'])[
                            'element_type'] not in
                                       devices]
                        target_gids = [x['target'] + base_idx for x in st if nest.GetDefaults(nest.GetStatus([x[
                                                                                                                  'target']])[
                                                                                                  0]['model'])[
                            'element_type'] not in
                                       devices]
                        weights = [x['weight'] for x in st if nest.GetDefaults(nest.GetStatus([x['target']])[0][
                                                                                   'model'])[
                            'element_type'] not in devices]
                        delays = [x['delay'] for x in st if nest.GetDefaults(nest.GetStatus([x['target']])[0][
                                                                                 'model'])[
                            'element_type'] not in devices]

                        receptors = [x['receptor'] for x in st if 'element_type' in nest.GetDefaults(nest.GetStatus([x['target']])[0][
                                                                                       'model']) and nest.GetDefaults(nest.GetStatus([x['target']])[0][
                                                                     'model'])['element_type'] not in devices]
                        # modify target receptors in cases where they are used:
                        # for iddx, n_rec in enumerate(receptors):
                        # 	# check if neuron accepts different receptors
                        # 	if nest.GetStatus([target_gids[iddx]])[0].has_key('rec_type'):
                        # 		rec_types = list(nest.GetStatus([target_gids[iddx]])[0]['rec_type'])
                        # 		receptors[iddx] = int(nest.GetStatus([target_gids[iddx]])[0]['rec_type'][n_rec])
                        syn_dicts = [{'synapse_model': list(np.repeat(copy_synapse_name, len(source_gids)))[iddx],
                                      'source': source_gids[iddx],
                                      'target': target_gids[iddx],
                                      'weight': weights[iddx],
                                      'delay': delays[iddx],
                                      'receptor_type': receptors[iddx]} for iddx in range(len(target_gids))]
                        nest.DataConnect(syn_dicts)
                        if progress:
                            progress_bar(float(nnn) / float(len(its)))
                logger.info("\tElapsed time: {0} s".format(str(time.time() - start)))

        def connect_decoders(network, parameters):
            """
            :param parameters:
            :return:
            """
            target_populations = parameters.decoding_pars.state_extractor.source_population
            copy_targets = [n + '_clone' for n in target_populations]
            parameters.decoding_pars.state_extractor.source_population = copy_targets
            network.connect_decoders(parameters.decoding_pars)

        clone_network = create_clone(self)
        connect_clone(self, clone_network, display)
        if devices:
            clone_network.connect_devices()
        if decoders:
            connect_decoders(clone_network, param_set)

        return clone_network

    @staticmethod
    def _parse_parameters(network_parameters, topology=None, spike_recorders=None, analog_recorders=None):
        """
        Unpack a compact set of parameters, from which all the relevant data is extracted.

        :param network_parameters: [dict or ParameterSet] - should have the following structure
        {'populations': list of strings, 'population_size': list of integers, 'neurons': list of neuron parameter
        sets or dictionaries}
        :param topology: None or list of dictionaries or ParameterSets
        :param spike_recorders: None or list of dictionaries or ParameterSets
        :param analog_recorders: None or list of dictionaries or ParameterSets
        :return: network ParameterSet
        """
        if isinstance(network_parameters, dict):
            network_parameters = ParameterSet(network_parameters)
        net_pars = {
            'n_populations': len(network_parameters.populations),
            'pop_names': network_parameters.populations,
            'n_neurons': network_parameters.population_size,
            'neuron_pars': network_parameters.neurons,
            'randomize': network_parameters.randomize,
            'topology': [False for _ in range(len(network_parameters.populations))],
            'topology_dict': [None for _ in range(len(network_parameters.populations))],
            'record_spikes': [False for _ in range(len(network_parameters.populations))],
            'spike_device_pars': [{} for _ in range(len(network_parameters.populations))],
            'record_analogs': [False for _ in range(len(network_parameters.populations))],
            'analog_device_pars': [None for _ in range(len(network_parameters.populations))]
        }
        if 'merged_populations' in network_parameters:
            net_pars.update({'merged_populations': network_parameters.merged_populations})

        if topology is not None:
            assert isinstance(topology, list) and isinstance(np.any(topology), dict) or isinstance(np.any(topology),
                                                                                                   ParameterSet), \
                "To add topology to the network, provide topology as a list of parameter dictionaries.."
            net_pars.update({'topology': [True for nn in range(len(network_parameters.populations)) if
                                          not empty(topology[nn])],
                             'topology_dict': topology})
        else:
            net_pars.update({'topology': [False for _ in range(len(network_parameters.populations))],
                             'topology_dict': [None for _ in range(len(network_parameters.populations))]})
        if spike_recorders is not None:
            assert isinstance(spike_recorders, list) and isinstance(np.any(spike_recorders), dict) or isinstance(
                np.any(spike_recorders), ParameterSet), \
                "To add spike recording devices to the network, provide a list of parameter dictionaries.."
            net_pars.update({'record_spikes': [True for nn in range(len(network_parameters.populations)) if
                                               not empty(spike_recorders[nn])],
                             'spike_device_pars': spike_recorders})
        else:
            net_pars.update({'record_spikes': [False for _ in range(len(network_parameters.populations))],
                             'spike_device_pars': [{} for _ in range(len(network_parameters.populations))]})

        if analog_recorders is not None:
            assert isinstance(analog_recorders, list) and isinstance(np.any(analog_recorders), dict) or isinstance(
                np.any(analog_recorders), ParameterSet), \
                "To add spike recording devices to the network, provide a list of parameter dictionaries.."
            net_pars.update({'record_analogs': [True for nn in range(len(network_parameters.populations)) if
                                               not empty(analog_recorders[nn])],
                             'analog_device_pars': analog_recorders})
        else:
            net_pars.update({'record_analogs': [False for _ in range(len(network_parameters.populations))],
                             'analog_device_pars': [{} for _ in range(len(network_parameters.populations))]})

        return ParameterSet(net_pars)

    def find_population(self, name):
        """
        Returns a (merged) population given its name. Raises KeyError if not found.

        :param name: [str] name of Population (e.g. 'E1')
        :return: Population object
        """
        try:
            return self.populations[name]
        except KeyError:
            try:
                return self.merged_populations[name]
            except KeyError:
                raise KeyError("Population named '{0}' not found!".format(name))

    def get_merged_population(self, population_names, populations=None, store=False):
        """
        Finds or creates a merged population from existing population objects. The name of the merged population
        is simply the concatenation of the population names.

        :param population_names: [list] names of the populations to merge
        :param populations: [list] list of Population objects to merge from, or None to search in self
        :param store: [bool] Whether to store the newly created merged population object in the snn object (self)
        :return:
        """
        merged_population_name = ''.join(population_names)

        try:
            return merged_population_name, self.merged_populations[merged_population_name]
        except (KeyError, AttributeError):
            if populations:
                populations_to_merge = [populations[p] for p in population_names]
            else:
                populations_to_merge = [self.populations[p] for p in population_names]

            merged_population = merge_subpopulations(populations_to_merge, name=merged_population_name)
            if store:
                self.merged_populations[merged_population_name] = merged_population

            return merged_population_name, merged_population

    def connect_recording_devices(self, spike_recorders=None, analog_recorders=None):
        """
        Connect recording devices to the populations, according to the parameter specifications

        NOTE: Should only be called once! Otherwise, connections are repeated, although a warning is printed.
        """
        if self._devices_connected:
            warnings.warn('Devices already connected to {}, connections will be repeated!'.format(self.name))

        if spike_recorders is not None:
            assert isinstance(spike_recorders, list) and isinstance(np.any(spike_recorders), dict) or isinstance(
                np.any(spike_recorders), ParameterSet), \
                "To add spike recording devices to the network, provide a list of parameter dictionaries.."
            assert len(spike_recorders) == len(self.populations), "Provide a list with recorders for each " \
                                                                  "population"
            self._record_spikes = [True for nn in range(len(self.populations)) if
                                   not empty(spike_recorders[nn])]
            self._spike_device_pars = spike_recorders

        if analog_recorders is not None:
            assert isinstance(analog_recorders, list) and isinstance(np.any(analog_recorders), dict) or isinstance(
                np.any(analog_recorders), ParameterSet), \
                "To add spike recording devices to the network, provide a list of parameter dictionaries.."
            assert len(spike_recorders) == len(self.populations), "Provide a list with recorders for each " \
                                                                  "population"
            self._record_analogs = [True for nn in range(len(self.populations)) if
                                    not empty(analog_recorders[nn])]
            self._analog_device_pars = analog_recorders

        logger.info("Connecting Devices: ")
        for n in range(self.n_populations):
            self._attach_recorder_to_population(n)

    def _attach_recorder_to_population(self, n):
        """

        :param n:
        :return:
        """
        pop = self.populations[self.population_names[n]]
        self.n_devices[n] = 0
        if self._record_spikes[n]:
            dev_dict = self._spike_device_pars[n].copy()
            self.device_type[n].append(dev_dict['model'])
            dev_dict['label'] += pop.name + '_' + dev_dict['model']
            self.device_names[n].append(dev_dict['label'])
            dev_gid = pop.record_spikes(parameters.extract_nestvalid_dict(dev_dict, param_type='device'))
            self.device_gids[n].append(dev_gid)
            self.n_devices[n] += 1
            logger.info("- Connecting {0!s} to {1!s}, with label {2!s} and id {3!s}".format(
                dev_dict['model'], self.population_names[n], dev_dict['label'], str(dev_gid)))

        if self._record_analogs[n]:
            if isinstance(self._record_analogs[n], bool):
                dev_dict = self._analog_device_pars[n].copy()
                self.device_type[n].append(dev_dict['model'])
                dev_dict['label'] += pop.name + '_' + dev_dict['model']
                self.device_names[n].append(dev_dict['label'])
                tmp_pop = self.populations[self.population_names[n]]
                if dev_dict['record_n'] != tmp_pop.size:
                    tmp = np.random.permutation(pop.size)[:dev_dict['record_n']]
                    ids = []
                    for i in tmp:
                        ids.append(tmp_pop.gids[i])
                else:
                    ids = None
                rec_pars_dict = parameters.extract_nestvalid_dict(dev_dict, param_type='device')
                dev_gid = tmp_pop.record_analog(ids=ids, record=dev_dict['record_from'],
                                                            rec_pars_dict=rec_pars_dict)
                self.device_gids[n].append(dev_gid)

                self.n_devices[n] += 1
                if (ids is not None) and (len(ids) == 1):
                    logger.info("- Connecting {0!s} to {1!s} [{2!s}], with label {3!s} and id {4!s}".format(
                        dev_dict['model'], pop.name, str(ids), dev_dict['label'], str(dev_gid)))
                elif ids is not None:
                    logger.info(
                        "- Connecting {0!s} to {1!s} [{2!s}-{3!s}], with label {4!s} and id {5!s}".format(
                            dev_dict['model'], pop.name, str(min(ids)), str(max(ids)),
                            dev_dict['label'], str(dev_gid)))
                else:
                    logger.info("- Connecting {0!s} to {1!s} [{2!s}], with label {3!s} and id {4!s}".format(
                        dev_dict['model'], pop.name, str('all'), dev_dict['label'],
                        str(dev_gid)))
            else:
                for nnn in range(int(self._record_analogs[n])):
                    dev_dict = self._analog_device_pars[n][nnn].copy()
                    self.device_type[n].append(dev_dict['model'])
                    dev_dict['label'] += pop.name + '_' + dev_dict['model']
                    self.device_names[n].append(dev_dict['label'])
                    if dev_dict['record_n'] != pop.size:
                        tmp = np.random.permutation(pop.size)[:dev_dict['record_n']]
                        ids = []
                        for i in tmp:
                            ids.append(self.populations[n].gids[i])
                    else:
                        ids = None

                    dev_pars = parameters.extract_nestvalid_dict(dev_dict, param_type='device')
                    dev_gid = self.populations[n].record_analog(dev_pars, ids=ids, record=dev_dict['record_from'])
                    self.device_gids[n].append(dev_gid)

                    self.n_devices[n] += 1
                    if len(ids) == 1:
                        logger.info("- Connecting {0!s} to {1!s} [{2!s}], with label {3!s} and id {4!s}".format(
                            dev_dict['model'], pop.name, str(ids), dev_dict['label'], str(dev_gid)))
                    else:
                        logger.info(
                            "- Connecting {0!s} to {1!s} [{2!s}-{3!s}], with label {4!s} and id {5!s}".format(
                                dev_dict['model'], pop.name, str(min(ids)), str(max(ids)),
                                dev_dict['label'], str(dev_gid)))

    def connect_state_extractors(self, initializer, encoder, input_mapper, stim_onset=0.1, stim_duration=None,
                                 stim_isi=None, variable_signal=False, to_memory=True):
        """
        Create and connect the state extractors to the individual populations in the network.

        :param initializer: [dict] dictionary with the state extractor parameters.
        :param encoder:
        :param input_mapper:
        :param stim_duration:
        :param stim_isi:
        :param stim_onset:
        :param variable_signal:
        :param to_memory:
        :return
        """
        self.state_extractor = SpikingExtractor(initializer, variable_signal, stim_duration, stim_isi,
                                                input_mapper.total_delay, encoder.resolution, to_memory)
        populations = copy.copy(self.populations)
        if hasattr(encoder, 'parrots') and isinstance(encoder.parrots, Population):
            populations[encoder.parrots.name] = encoder.parrots
        self.state_extractor.connect_populations(self, populations, stim_onset)

    def create_decoder(self, initializer, rng=None):
        """
        Create and store Decoder object.

        :param initializer: [dict] dictionary with the decoding layer parameters
        :param rng: RNG state
        :return
        """
        self.decoders = Decoder(initializer, rng=rng)

    def flush_records(self):
        """
        Delete all data from all devices connected to the network
        """
        if not empty(self.device_names):
            logger.info("\nClearing device data: ")
        devices = list(itertools.chain.from_iterable(self.device_names))

        for idx, n in enumerate(list(itertools.chain.from_iterable(self.device_gids))):
            logger.info(" - {0} {1}".format(devices[idx], str(n)))
            nest.SetStatus(n, {'n_events': 0})
            if nest.GetStatus(n)[0]['to_file']:
                tools.utils.data_handling.remove_files(nest.GetStatus(n)[0]['filenames'])

    def copy(self):
        """
        Returns a copy of the entire network_architect object. Doesn't create new NEST objects.

        :return:
        """
        return copy.deepcopy(self)

    def mirror(self, copy_net=None, from_main=True, to_main=False):
        """
        Returns a Network object equal to self and either connected
        to main Network or receiving connections from main Network
        :return:
        """
        if copy_net is None:
            copy_net = self.copy()
            cn = None
        else:
            assert isinstance(copy_net, Network), "copy_net must be Network object"
            cn = 1.

        if to_main:
            logger.info("Connecting CopyNetwork: ")
            device_models = ['spike_detector', 'spike_generator', 'multimeter']
            for pop_idx, pop_obj in enumerate(copy_net.populations):
                original_gid_range = [min(self.populations[pop_idx].gids),
                                      max(self.populations[pop_idx].gids)]
                copy_gid_range = [min(pop_obj.gids), max(pop_obj.gids)]

                start = time.time()
                logger.info("    - {0}, {1}".format(copy_net.population_names[pop_idx], self.population_names[
                    pop_idx]))

                for n_neuron in range(self.n_neurons[pop_idx]):
                    src_gid = self.populations[pop_idx].gids[n_neuron]
                    tget_gid = pop_obj.gids[n_neuron]

                    conns = nest.GetConnections(source=[src_gid])

                    # for memory conservation, iterate:
                    iterate_steps = 100
                    its = np.arange(0, len(conns), iterate_steps).astype(int)

                    for nnn, it in enumerate(its):
                        if nnn < len(its) - 1:
                            conn = conns[it:its[nnn + 1]]
                            st = nest.GetStatus(conn)

                        target_gids = [x['target'] for x in st if
                                       nest.GetStatus([x['target']])[0]['model'] not in device_models]
                        weights = [x['weight'] for x in st if
                                   nest.GetStatus([x['target']])[0]['model'] not in device_models]
                        delays = [x['delay'] for x in st if
                                  nest.GetStatus([x['target']])[0]['model'] not in device_models]
                        models = [x['synapse_model'] for x in st if
                                  nest.GetStatus([x['target']])[0]['model'] not in device_models]
                        receptors = [x['receptor'] for x in st if
                                     nest.GetStatus([x['target']])[0]['model'] not in device_models]

                        tgets = [x + (copy_gid_range[0] - original_gid_range[0]) for x in target_gids]
                        syn_dicts = [{'synapsemodel': models[iddx], 'source': tget_gid,
                                      'target': tgets[iddx], 'weight': weights[iddx],
                                      'delay': delays[iddx], 'receptor_type': receptors[iddx]} for iddx in range(len(
                            target_gids))]
                        nest.DataConnect(syn_dicts)
                logger.info("Elapsed Time: {0}".format(str(time.time() - start)))

        elif from_main:
            logger.info("\nConnecting CopyNetwork: ")
            device_models = ['spike_detector', 'spike_generator', 'multimeter']
            for pop_idx, pop_obj in enumerate(copy_net.populations):
                original_gid_range = [min(self.populations[pop_idx].gids),
                                      max(self.populations[pop_idx].gids)]
                copy_gid_range = [min(pop_obj.gids), max(pop_obj.gids)]

                start = time.time()
                logger.info(
                    "\t    - {0}, {1}".format(copy_net.population_names[pop_idx], self.population_names[pop_idx]))

                for n_neuron in range(self.n_neurons[pop_idx]):
                    src_gid = self.populations[pop_idx].gids[n_neuron]
                    tget_gid = pop_obj.gids[n_neuron]

                    conns = nest.GetConnections(target=[src_gid])

                    # for memory conservation, iterate:
                    iterate_steps = 100
                    its = np.arange(0, len(conns), iterate_steps).astype(int)

                    for nnn, it in enumerate(its):
                        if nnn < len(its) - 1:
                            conn = conns[it:its[nnn + 1]]
                            st = nest.GetStatus(conn)
                            source_gids = [x['source'] for x in st if
                                           nest.GetStatus([x['source']])[0]['model'] not in device_models]
                            weights = [x['weight'] for x in st if
                                       nest.GetStatus([x['target']])[0]['model'] not in device_models]
                            delays = [x['delay'] for x in st if
                                      nest.GetStatus([x['target']])[0]['model'] not in device_models]
                            models = [x['synapse_model'] for x in st if
                                      nest.GetStatus([x['target']])[0]['model'] not in device_models]
                            receptors = [x['receptor'] for x in st if
                                         nest.GetStatus([x['target']])[0]['model'] not in device_models]

                            sources = [x + (copy_gid_range[0] - original_gid_range[0]) for x in source_gids]
                            syn_dicts = [{'synapsemodel': models[iddx], 'source': sources[iddx],
                                          'target': tget_gid, 'weight': weights[iddx],
                                          'delay': delays[iddx], 'receptor_type': receptors[iddx]} for iddx in
                                         range(len(
                                             source_gids))]
                            nest.DataConnect(syn_dicts)
                logger.info("Elapsed Time: {0}".format(str(time.time() - start)))
        if cn is None:
            return copy_net

    @staticmethod
    def next_onset_time():
        return nest.GetKernelStatus('time') + nest.GetKernelStatus('resolution')

    def simulate(self, t=10, reset_devices=False, reset_network=False):
        """
        Simulate network_architect

        :param t: total time
        :param reset_devices:
        :param reset_network:
        :return:
        """
        nest.Simulate(t)
        if reset_devices:
            self.flush_records()
        if reset_network:
            nest.ResetNetwork()

    @staticmethod
    def _continuous_batch_processing(stim_seq):
        return isinstance(stim_seq[0], signals.AnalogSignalList)

    def process_batch(self, batch_label, encoder, stim_seq, stim_seq_signal=None, prev_stim_info=None, extract=False):
        """
        Simulates a stimulus sequence in the NEST kernel. The simulation time is strictly limited to the
        signal duration, all delays (encoding, decoding) are ignored.
        :param batch_label:
        :param encoder:
        :param stim_seq: [tuple] the full sequence to be processed in this batch or a
                         a generator yielding each stimulus individually
        :param stim_seq_signal: [AnalogSignalList] input signal to process (complete batch)
        :param prev_stim_info: [tuple] stimulus info
        :param extract: [bool] extract the sampled activity at each step (sequential)
        :return: [list] list of tuples containing the stimulus times (start, stop, interval)
        """
        logger.info("")
        logger.info("################## Simulating batch {}".format(batch_label))
        assert isinstance(stim_seq, tuple)

        # sequential ######################################
        if not self._continuous_batch_processing(stim_seq):
            stim_seq_gen = stim_seq[0]
            stim_info = [] if prev_stim_info is None else [prev_stim_info]

            cnt = 0 if prev_stim_info is None else 1
            for stim_seq_signal, stim_info_tuple in stim_seq_gen:
                encoder.update_state(stim_seq_signal)
                self.process_token(stim_seq_signal)

                if cnt > 0 and extract:
                    prev_stim_info = stim_info[-1]
                    self.state_extractor.extract_global_states([prev_stim_info], carry_begin=True)

                stim_info.append(stim_info_tuple)
                cnt += 1
        # continuous ######################################
        else:
            stim_info = None  # just to return something, not important in this case
            if stim_seq_signal is None:
                stim_seq_signal, stim_info = stim_seq
                encoder.update_state(stim_seq_signal)
            self.process_token(stim_seq_signal)

        return stim_info

    def process_token(self, stim_seq_signal):
        """
        Process a single token from a batch, for which the NEST generators have been updated before.
        """
        # simulate, from current kernel time until the end of the signal
        sim_time = stim_seq_signal.t_stop
        sim_time -= max(nest.GetKernelStatus('time') + nest.GetKernelStatus('resolution'), stim_seq_signal.t_start)
        self.simulate(sim_time)

    def _extract_and_train_batch(self, _label, _stim_info, _target_outputs, _total_delay,
                                 extract_stepwise=False, vocabulary=None):
        """
        Simulates delays for a correct state extraction, followed by training of the decoders.
        :return:
        """
        logger.info("Simulating delays from batch {}".format(_label))
        nest.Simulate(_total_delay + nest.GetKernelStatus('resolution'))

        # post-processing & training
        logger.info("Post-processing batch {}".format(_label))

        if not extract_stepwise:
            # self.state_extractor.extract_global_states([_stim_info[-1]], batch_end=True, carry_begin=True)
            self.state_extractor.extract_global_states(_stim_info, batch_end=True, carry_begin=True)
        else:
            self.state_extractor.extract_global_states(_stim_info, batch_end=True)

        self.state_extractor.compile_state_matrices(_label)
        self.state_extractor.report()
        self.decoders.train(self.state_extractor, _label, _target_outputs, _stim_info, vocabulary=vocabulary)
        self.state_extractor.flush_states()
        logger.info("Finished training batch {}".format(_label))

    def train(self, data_batch, n_epochs, sequencer, encoder, onset_time=None, total_delay=0., intervals=None,
              symbolic=False, continuous=True, verbose=True, save=False):
        """
        Train the network on the whole dataset

        TODO handle intervals properly, currently parameter doesn't exist and it's always set to None

        :param data_batch:
        :param n_epochs:
        :param sequencer:
        :param encoder:
        :param onset_time:
        :param total_delay:
        :param intervals:
        :param symbolic:
        :param continuous:
        :param verbose:
        :param save:
        :return:
        """
        assert n_epochs > 0
        assert "inputs", "decoder_outputs" in data_batch.keys()

        inputs = data_batch['inputs']
        targets = data_batch['decoder_outputs']
        n_batches = len(inputs)
        # for symbolic sequences, we need to pass the full vocabulary to the training routines in order to have
        # access to all possible labels, even if only a subset of these are presented in each batch
        vocabulary = sequencer.vocabulary if symbolic else None
        # vocabulary = None

        if onset_time is None:
            onset_time = self.next_onset_time()

        states = []
        outputs = []
        losses = []
        dec_loss = []
        dec_outputs = []
        dec_targets = []
        perf = None

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                batch_label = 'train_batch={}_epoch={}'.format(batch+1, epoch+1)
                if isinstance(sequencer, preprocessing.ImageFrontend):
                    batch_input = sequencer.draw_stimulus_sequence(inputs[batch], as_array=False, unfold=True,
                                                                   onset_time=onset_time, continuous=continuous,
                                                                   intervals=intervals, verbose=False)
                else:
                    batch_input = sequencer.draw_stimulus_sequence(inputs[batch], onset_time=onset_time, verbose=False,
                                                                   intervals=intervals, continuous=continuous)

                batch_states = self.fit(batch_label, n_batches * n_epochs, stim_seq=batch_input, encoder=encoder,
                                        target_outputs=targets[batch], total_delay=total_delay, save=save,
                                        vocabulary=vocabulary)
                onset_time = self.next_onset_time()

            self.decoders.validate(batch_label, symbolic, vocabulary)
            if epoch == n_epochs - 1:
                dec_loss = self.decoders.validation_accuracy
                perf = self.decoder_accuracy(output_parsing="k-WTA" if symbolic else None, symbolic_task=symbolic,
                                             vocabulary=vocabulary)

            if epoch == 0 or epoch == n_epochs-1:
                # state here is from the last processed batch
                states.append(batch_states)
                outputs.append([r.output for r in self.decoders.readouts])
                outs, tgts = self.decoders.retrieve_outputs()
                dec_outputs.append(outs)
                dec_targets.append(tgts)

        return {'losses': losses, 'states': states, 'outputs': outputs, 'decoder_loss': dec_loss,
                'decoder_accuracy': perf, 'decoder_outputs': dec_outputs, 'decoder_targets': dec_targets}

    def test(self, data_batch, sequencer, encoder, onset_time=None, total_delay=0., intervals=None,
             symbolic=False, continuous=True, output_parsing='k-WTA', verbose=True, save=False):
        """
        Process test set
        """
        inputs = data_batch['inputs']
        n_batches = len(inputs)
        targets = data_batch['decoder_outputs']
        vocabulary = sequencer.vocabulary if symbolic else None

        if onset_time is None:
            onset_time = self.next_onset_time()

        for batch in range(n_batches):
            batch_label = 'test_batch={}'.format(batch+1)

            if isinstance(sequencer, preprocessing.ImageFrontend):
                batch_input = sequencer.draw_stimulus_sequence(inputs[batch], as_array=False, unfold=True,
                                                               onset_time=onset_time, continuous=continuous,
                                                               intervals=intervals, verbose=False)
            else:
                batch_input = sequencer.draw_stimulus_sequence(inputs[batch], onset_time=onset_time, verbose=False,
                                                               intervals=intervals, continuous=continuous)

            batch_states = self.predict(batch_label, encoder, stim_seq=batch_input, target_outputs=targets[batch],
                                        total_delay=total_delay)
            onset_time = self.next_onset_time()

        perf = self.decoder_accuracy(output_parsing=output_parsing, symbolic_task=symbolic, vocabulary=vocabulary)
        outs, tgts = self.decoders.retrieve_outputs()

        return {'states': batch_states, 'decoder_accuracy': perf, 'outputs': self.decoders.readouts[0].output,
                'decoder_outputs': outs, 'decoder_targets': tgts}

    def fit(self, batch_label, n_batches, stim_seq, encoder, target_outputs, total_delay=0., save=True,
            vocabulary=None):
        """
        Training on a single batch. The very first training batch is initially just simulated, the actual extraction and
        training begins with the second batch: initialize the current batch, simulate the (decoding) delays
        for the previous batch, extract its states and train readouts, and finally simulate the rest of the current
        batch.

        :return: state
        """
        assert isinstance(target_outputs, list)
        assert isinstance(stim_seq, tuple) #and isinstance(stim_seq[0], types.GeneratorType)

        if not self.decoders.initialized:
            self.decoders.connect(self.state_extractor, target_outputs)

        extract_stepwise = not self._continuous_batch_processing(stim_seq)

        # first batch, just simulate sequence
        if self.batch_cnt == 0:
            stim_info = self.process_batch(batch_label, encoder, stim_seq, extract=extract_stepwise)
            if n_batches == 1:
                self._extract_and_train_batch(batch_label, stim_info, target_outputs, total_delay,
                                              extract_stepwise=False, vocabulary=vocabulary)
        # post-process previous batch, and simulate current one
        else:
            logger.info('Initializing training batch {}'.format(batch_label))

            if not self._continuous_batch_processing(stim_seq):
                stim_seq_signal, stim_info = next(stim_seq[0])
                encoder.update_state(stim_seq_signal)  # update NEST generators
                # post-process previous batch
                self._extract_and_train_batch(self.prev_batch_data['label'],
                                              self.prev_batch_data['stim_info'],
                                              self.prev_batch_data['target_outputs'], total_delay,
                                              extract_stepwise=extract_stepwise, vocabulary=vocabulary)
                self.process_token(stim_seq_signal)
                # simulate current batch
                stim_info = self.process_batch(batch_label, encoder, stim_seq, extract=True, prev_stim_info=stim_info)
            else:
                stim_seq_signal, stim_info = stim_seq
                encoder.update_state(stim_seq_signal)  # update NEST generators
                # post-process previous batch
                self._extract_and_train_batch(self.prev_batch_data['label'],
                                              self.prev_batch_data['stim_info'],
                                              self.prev_batch_data['target_outputs'], total_delay,
                                              extract_stepwise=extract_stepwise, vocabulary=vocabulary)
                # simulate current batch
                self.process_batch(batch_label, encoder, stim_seq, stim_seq_signal, stim_info)

            # if last batch, post-process here directly
            if self.batch_cnt == n_batches - 1:
                self._extract_and_train_batch(batch_label, stim_info, target_outputs, total_delay,
                                              extract_stepwise=extract_stepwise, vocabulary=vocabulary)
                if save:
                    self.decoders.save_training_data()  # save readout data after training

        # store data of current batch for the next iteration
        self.prev_batch_data['label'] = batch_label
        self.prev_batch_data['stim_info'] = stim_info
        self.prev_batch_data['target_outputs'] = target_outputs
        self.batch_cnt += 1

        # return all the extracted states for each readout in particular
        states = []
        for ext_label in list(self.state_extractor.get_labels()):
            states.append(self.state_extractor.get_state_matrix(ext_label))
        return states

    def predict(self, set_label, encoder, stim_seq, target_outputs, total_delay=0.):
        """

        :param set_label:
        :param stim_seq:
        :param encoder:
        :param target_outputs:
        :param total_delay:
        :return:
        """
        logger.info("Simulating test set {}".format(set_label))
        assert isinstance(target_outputs, list)
        assert isinstance(stim_seq, tuple), "stim_seq must be a tuple (DynamicEmbeddings/AnalogSignalList, stim_info)"

        self.state_extractor.flush_states(carry=True)  # need a clean extractor here
        carry_begin = False
        if not self._continuous_batch_processing(stim_seq):
            # simulate one step here (to be able to reinitialize the state extractors), rest is done in process_batch
            stim_seq_signal, stim_info_first = next(stim_seq[0])
            self.state_extractor.reinitialize_extractors(stim_seq_signal[0].t_start)
            self.process_token(stim_seq_signal)

            # simulate rest of generator
            stim_info = self.process_batch(set_label, encoder, stim_seq, extract=True, prev_stim_info=stim_info_first)
            self.simulate(total_delay + nest.GetKernelStatus('resolution'))
            carry_begin = True
        else:
            # reset extractors due to a discontinuity in the sampling timeline (delays) between training and prediction
            self.state_extractor.reinitialize_extractors(stim_seq[0].t_start)
            stim_info = self.process_batch(set_label, encoder, stim_seq)

        self.simulate(total_delay + nest.GetKernelStatus('resolution'))
        self.state_extractor.extract_global_states(stim_info, batch_end=True, carry_begin=carry_begin)
        self.state_extractor.compile_state_matrices(set_label, force_save=True)
        self.state_extractor.report()

        # run predictions when dataset has been fully simulated and states extracted
        self.decoders.predict(self.state_extractor, target_outputs, stim_info)

        # return the state for each readout in particular
        states = []
        # for readout in self.decoders.readouts:
        #     states.append(self.state_extractor.get_state_matrix(readout.extractor))
        for ext_label in list(self.state_extractor.get_labels()):
            states.append(self.state_extractor.get_state_matrix(ext_label))

        self.state_extractor.flush_states()
        logger.info("Finished test set {}".format(set_label))
        return states

    def decoder_accuracy(self, output_parsing='k-WTA', symbolic_task=True, vocabulary=None):
        """
        Evaluate decoder accuracy
        :param output_parsing: None, 'k-WTA', 'threshold', 'softmax'
        :param symbolic_task: bool
        :return:
        """
        mse_only = not symbolic_task  # if the task is symbolic, all evaluations should be carried
        return self.decoders.evaluate(process_output_method=output_parsing, symbolic=symbolic_task,
                                      mse_only=mse_only, vocabulary=vocabulary)

    def validate(self, stim_seq, encoder, target_outputs, total_delay):
        """

        :return:
        """
        self.predict('validation', stim_seq, encoder, target_outputs, total_delay)
        self.decoders.validate()

    def extract_population_activity(self, t_start=None, t_stop=None):
        """
        Iterate through the populations in the network_architect, verify which recording devices have been connected
        to them and extract this data. The activity is then converted in SpikeList or AnalogList objects and attached
        to the properties of the corresponding population.
        To merge the data from multiple populations see extract_network_activity()
        """
        if not empty(self.device_names):
            logger.info("\nExtracting and storing recorded activity from devices:")

        sim_res = nest.GetKernelStatus('resolution')
        for pop_name, pop_obj in {**self.populations, **self.merged_populations}.items():
            if isinstance(pop_obj, list):
                for pop_idx, pop in enumerate(pop_obj):
                    if pop.attached_devices:
                        logger.info("- Population {0}".format(pop.name))
                        for nnn in pop.attached_devices:
                            if nest.GetStatus(nnn)[0]['to_memory']:
                                # initialize load_activity with device gid
                                pop.load_activity(nnn, time_shift=sim_res, t_start=t_start, t_stop=t_stop)
                            elif nest.GetStatus(nnn)[0]['to_file']:
                                # initialize load_activity with file paths
                                pop.load_activity(list(nest.GetStatus(nnn)[0]['filenames']), time_shift=sim_res,
                                                  t_start=t_start, t_stop=t_stop)
            else:
                if pop_obj.attached_devices:
                    logger.info("- Population {0}".format(pop_obj.name))
                    for device_gid in pop_obj.attached_devices:
                        if nest.GetStatus(device_gid)[0]['to_memory']:
                            # initialize load_activity with device gid
                            pop_obj.load_activity(device_gid, time_shift=sim_res, t_start=t_start, t_stop=t_stop)
                        elif nest.GetStatus(device_gid)[0]['to_file']:
                            pop_obj.load_activity(list(nest.GetStatus(device_gid)[0]['filenames']), time_shift=sim_res,
                                                  t_start=t_start, t_stop=t_stop)

    def find_state_matrix(self, pop_name, extractor_label):
        """
        Finds and returns a given state matrix for a population.

        :param pop_name:
        :param extractor_label:
        :return: [StateMatrix] object or None, if population or extractor not found
        """
        try:
            pop = self.find_population(pop_name)
            try:
                return pop.state_matrices[extractor_label]
            except KeyError:
                logger.error('State extractor [{}] for population {} not found.'.format(extractor_label, pop_name))
        except KeyError:
            logger.error('Population {} not found.'.format(pop_name))
        return None

    # TODO implement
    def save_state(self):
        pass

    # TODO implement
    def restore_state(self):
        pass

    def extract_activity(self, t_start=None, t_stop=None, flush=True):
        """
        Extract activity recorded from all devices attached to the network.
        :param t_start:
        :param t_stop:
        :param flush: remove activity from lower-level Population objects after copying to Network
        """
        for n, (name, pop) in enumerate(self.populations.items()):
            # if isinstance(pop, list) and self.n_devices[n]:
            #     for nn in range(len(pop)):
            #         if empty(pop[nn].spiking_activity) and empty(pop[nn].analog_activity):
            #             self.extract_population_activity(t_start=t_start, t_stop=t_stop)
            #         self.spiking_activity[n].append(self.populations[n][nn].spiking_activity)
            #         self.analog_activity[n].append(self.populations[n][nn].analog_activity)
            #         if flush:
            #             self.populations[n][nn].spiking_activity = []
            #             self.populations[n][nn].analog_activity = []
            if self.n_devices[n]:
                if empty(pop.spiking_activity) and empty(pop.analog_activity):
                    self.extract_population_activity(t_start=t_start, t_stop=t_stop)
                self.spiking_activity[n] = pop.spiking_activity
                self.analog_activity[n] = pop.analog_activity
                if flush:
                    pop.spiking_activity = []
                    pop.analog_activity = []

    def report(self):
        """
        Print a description of the system
        """
        logger.info("========================================================")
        logger.info(" {0!s} architecture ({1!s}-simulated):".format(self.name, self.simulator))
        logger.info("--------------------------------------------------------")
        logger.info("- {0!s} Populations: {1!s}".format(len(self.population_names), self.population_names))
        logger.info("- Size: {0!s} {1!s}".format(np.sum(self.n_neurons), self.n_neurons))
        logger.info("- Neuron models: {0!s}".format([n['model'] for n in
                                                     iterate_obj_list(self._parameters.neuron_pars)]))
        logger.info("- {0!s} Devices connected: {1!s}".format(self.n_devices, self.device_names))
        for n_pop, pop in enumerate(self.population_names):
            logger.info("    {0!s}: {1!s}, {2!s}".format(pop, self.device_names[n_pop], self.device_type[n_pop]))

    @staticmethod
    def print(depth=2):
        """
        Prints entire network structure, as setup in NEST
        :param depth:
        :return:
        """
        logger.info("Network Structure: \n{0!s}".format(nest.PrintNetwork(depth)))

    def get_neuron_models(self):
        """
        Returns a dictionary of unique neuron model names (key) and associated neuron ids
        :return:
        """
        models = np.unique([nest.GetStatus([n], 'model')[0].name for pop in self.populations for n in pop.gids])
        neurons = {k: [] for k in models}
        for pop in self.populations:
            for n in pop.gids:
                neurons[pop.name].append(n)
        return neurons

    # TODO - remove profiling methods - should be implemented externally
    def noise_driven_dynamics(self, input_rate, input_density, analysis_parameters):
        """
        Probe the dynamics of SNNs: drive the network with Poissonian input at a constant
        input_rate, connected to all the neurons with a pairwise_bernoulli connection profile with density
        input_density and report the statistics of network responses, according to the depth of analysis specified
        in the analysis parameters
        :param input_rate: firing rate of Poissonian input
        :param input_density: connection density
        :param analysis_parameters: [dict or ParameterSet]
        :return results:
        """
        pass

    def neuron_transferfunction(self, gid=None, input_type='rate', input_range=None, input_weights=None,
                                total_time=1000., step=100., restore=False, plot=True, display=True, save=False):
        """
        Numerically determine a single neuron's rate or current transfer function for the selected neurons or for a
        representative set
        :param gid: int, list or tuple - id of the neuron to record from (if None, one random neuron will be used)
        :param input_type: str - 'rate' or 'current' input
        :param input_range: list or tuple - range of input firing rates (Hz) or currents (pA)
        :param input_weights: list or tuple - synaptic weights for E/I spike input (valid only if input_type='rate')
        :param total_time: [ms] float
        :param step: [ms] duration of a single I/O measurement
        :param restore: bool - save and restore network state after analysis
        :param plot: bool - plot the main results
        :param display: bool - show plots and detailed outputs
        :param save: store results in provided path
        :return results: dict
        """
        # TODO disconnect and/or restore, record from multiple gids (split analysis)
        if self._is_connected:
            logger.warning("Single neuron analyses in connected populations may lead to incorrect results..")
        if gid is None:
            gid = [np.random.choice(v) for k, v in self.get_neuron_models().items()]
        else:
            assert isinstance(gid, int) or isinstance(gid, list) or isinstance(gid, tuple), "Provide gids as int, " \
                                                                                           "list or tuple"
            if isinstance(gid, int):
                gid = [gid]
        assert isinstance(input_range, list) or isinstance(input_range, tuple), "Provide the range of input rates as a list " \
                                                                        "or tuple (min, max)"
        dt = nest.GetKernelStatus()['resolution']
        t_start = nest.GetKernelStatus()['time']
        t_stop = t_start + total_time
        times = list(np.arange(t_start, t_stop, step))[1:]
        input_amplitudes = list(np.linspace(input_range[0], input_range[1], len(times)))

        if input_type == 'rate':
            assert input_weights is not None, "No synaptic weight provided"
            gen = nest.Create('inhomogeneous_poisson_generator', len(input_weights))
            nest.SetStatus(gen, {'start': t_start+dt, 'stop': t_stop, 'rate_times': times,
                                 'rate_values': input_amplitudes})
            recordables = [str(x) for x in nest.GetStatus(gid)[0]['recordables']]
            for idx_in, w_in in enumerate(input_weights):
                nest.Connect([gen[idx_in]], gid, syn_spec={'weight': w_in})

        elif input_type == 'current':
            gen = nest.Create('step_current_generator')
            nest.SetStatus(gen, {'start': t_start+dt, 'stop': t_stop, 'amplitude_times': times,
                                 'amplitude_values': input_amplitudes})
            recordables = ['V_m']
            nest.Connect(gen, gid)

        else:
            raise NotImplementedError("input_type {} not implemented yet".format(input_type))

        mm = nest.Create('multimeter', 1, {'record_from': recordables, 'start': t_start+dt, 'stop': t_stop+dt})
        nest.Connect(mm, gid)
        spks = nest.Create('spike_detector', 1, {'start': t_start+dt, 'stop': t_stop+dt})
        nest.Connect(gid, spks)

        nest.Simulate(total_time+dt)

        results = {}
        if input_type == 'current':
            results = single_neuron_dcresponse(neuron_gid=gid[0], input_amplitudes=input_amplitudes, input_times=times,
                                        spike_list=signals.convert_activity(spks),
                                        analogs=signals.convert_activity(mm),
                                        plot=plot, display=display, save=save)
        elif input_type == 'rate':
            results = single_neuron_ratetransfer(neuron_gid=gid[0], input_amplitudes=input_amplitudes,
                                                 input_times=times, spike_list=signals.convert_activity(spks),
                                                 analogs=signals.convert_activity(mm), recordables=recordables,
                                                 plot=plot, display=display, save=save)

        return results

    def population_transferfunction(self, range_=None):
        """
        Numerically determine the transfer function for the network as a whole and the individual populations
        :param range_:
        :return:
        """
        pass

    def self_sustaining_activity(self):
        """

        :return:
        """
        pass

    def perturbation_analysis(self):
        """

        :return:
        """
        pass

    def intrinsic_timescale(self):
        """

        :return:
        """
        pass
