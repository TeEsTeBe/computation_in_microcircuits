"""
========================================================================================================================
Network Architect Module
========================================================================================================================

Classes
-------------
Population                - Population object used to handle the simulated neuronal populations
Network                   - Abstract Network class

Functions
-------------
merge_subpopulations

========================================================================================================================

"""

# other imports
import itertools
import numpy as np
import importlib

nest = None if importlib.util.find_spec("nest") is None else importlib.import_module("nest")

# internal imports
from fna import tools
from fna.tools import parameters, signals
from fna.tools.utils.operations import empty
from abc import ABC, abstractmethod

logger = tools.utils.logger.get_logger(__name__)


def merge_subpopulations(sub_populations, name=''):
    """
    Combine sub-populations into a single Population object.

    :param sub_populations: [list] - of Population objects to merge
    :param name: [str] - name of new population
    :return: new Population object
    """
    assert sub_populations, "No sub-populations to merge provided..."
    gids_list = [list(x.gids) for x in sub_populations]
    gids = list(itertools.chain.from_iterable(gids_list))

    subpop_names = [x.name for x in sub_populations]
    if empty(name):
        name = ''.join(subpop_names)

    pop_dict = {'pop_names': name, 'n_neurons': len(gids), 'gids': gids, 'is_subpop': False}

    if all([x.topology for x in sub_populations]):
        positions = [list(x.topology_dict['positions']) for x in sub_populations]
        positions = list(itertools.chain.from_iterable(positions))

        elements = [x.topology_dict['elements'] for x in sub_populations]

        layer_ids = [x.layer_gid for x in sub_populations]

        tp_dict = {'elements': elements, 'positions': positions,
                   'edge_wrap': all(bool('edge_wrap' in x.topology_dict and x.topology_dict['edge_wrap'])
                                    for x in sub_populations)}

        pop_dict.update({'topology': True, 'layer_gid': layer_ids, 'topology_dict': tp_dict})
    else:
        pop_dict.update({'topology': False})

    new_population = Population(parameters.ParameterSet(pop_dict))

    new_population.children = sub_populations
    new_population.is_merged = True

    return new_population


class Population(object):
    """
    Population object used to handle each of the simulated neuronal populations
    Contains the parameters of each population (name, size, topology, gids);
    allows recording devices to be connected to the population (all neurons or a
    randomly sampled subset), according to the parameters specifications (see record_spikes
    and record_analog)
    After the simulation is terminated or a short period of simulation is run, the records
    can be deleted with flush_records() or converted into a SpikeList object, which
    will remain a property of the population object (using the function load_activity())

    Input:
        - pop_set -> ParameterSet for this population
    """

    def __init__(self, pop_set):
        self.name = pop_set.pop_names
        self.size = pop_set.n_neurons
        self.topology = pop_set.topology
        if self.topology:
            self.layer_gid = pop_set.layer_gid
            self.topology_dict = pop_set.topology_dict
        self.gids = pop_set.gids
        self.spiking_activity = []
        self.analog_activity = []
        self.is_subpop = pop_set.is_subpop
        self.attached_devices = []
        self.attached_device_names = []
        self.analog_activity_names = []
        self.seq_id = None  # [int] sequence id of population according to the order defined in the parameters file
        self.is_merged = False  # [bool] whether this population is a merged population (e.g. EI)
        self.children = None  # [list] populations (objects) this merged one is composed of

        self.state_matrices = {}

    def initialize_states(self, var_name, randomization_function, success_threshold=500, **function_parameters):
        """
        Randomize the initial value of a specified variable, property of the neurons in this population
        applying the function specified, with the corresponding parameters
        :param var_name: [str] - name of the variable to randomize
        :param randomization_function: [function] - randomization function to generate the values
        :param success_threshold: [int] - maximum number of attempts (sometimes required)
        :param function_parameters: extra parameters of the function

        example:
        --------
        >> n.randomize_initial_states('V_m', randomization_function=np.random.uniform,
         low=-70., high=-55.)
        """
        # TODO - multi-threaded SetStatus!
        # for thread in np.arange(nest.GetKernelStatus('local_num_threads')):
        #     local_nodes = nest.GetNodes([0], {'model': self.name, 'thread': thread}, local_only=True)[0]

        assert var_name in list(nest.GetStatus(self.gids)[0].keys()), "Variable name not in object properties"
        logger.info("- Randomizing {0} state in Population {1}".format(str(var_name), str(self.name)))
        try:
            nest.SetStatus(self.gids, var_name, randomization_function(size=len(self.gids), **function_parameters))
        except Exception as e:
            logger.info('WARNING - Chosen randomization function might contain an error: {}'.format(str(e)))
            for n_neuron in self.gids:
                success = False
                iter = 0
                # depending on the randomization function, we might need to try this multiple times
                while not success and iter < success_threshold:
                    try:
                        nest.SetStatus([n_neuron], var_name, randomization_function(size=1, **function_parameters))
                        success = True
                    except:
                        iter += 1
                if not success:
                    raise ValueError('Randomization function failed to generate correct values for NEST after '
                                     '500 attempts. If you are certain it is correct, just raise the threshold.')

    def record_spikes(self, rec_pars_dict, ids=None, label=None):
        """
        Connect a spike detector to this population.

        :param rec_pars_dict: common dictionary with recording device parameters
        :param ids: neuron ids to connect to (if None, all neurons are connected)
        :param label: device label to be registered in NEST
        """
        if label is not None:
            nest.CopyModel('spike_detector', label)
            det = nest.Create(label, 1, rec_pars_dict)
            self.attached_device_names.append(label)
        else:
            det = nest.Create('spike_detector', 1, rec_pars_dict)
        self.attached_devices.append(det)
        logger.info("  - Attaching spike_detector with gid [{}] to population {}".format(det, self.name))
        if ids is None:
            nest.Connect(self.gids, det, 'all_to_all')
        else:
            nest.Connect(list(ids), det, 'all_to_all')
        return det

    def record_analog(self, rec_pars_dict, ids=None, record=['V_m'], label=None):
        """
        Connect a multimeter to neurons with gid = ids and record the specified variables.

        :param rec_pars_dict: common dictionary with recording device parameters
        :param ids: neuron ids to connect (if None all neurons will be connected)
        :param record: recordable variable(s) - as list
        """
        st = nest.GetStatus([self.gids[np.random.randint(self.size)]])[0]
        if 'recordables' in st:
            assert np.mean([x in list(nest.GetStatus([self.gids[np.random.randint(self.size)]], 'recordables')[0])
                            for x
                            in record]) == 1., "Incorrect setting. Record should only contain recordable instances of " \
                                               "the current neuron model (check 'recordables')"
        assert np.mean([x in list(nest.GetDefaults('multimeter').keys()) for x in rec_pars_dict.keys()]), "Provided " \
                                                                                                          "dictionary is " \
                                                                                                          "inconsistent " \
                                                                                                          "with " \
                                                                                                          "multimeter " \
                                                                                                          "dictionary"
        if label is not None:
            nest.CopyModel('multimeter', label)
            mm = nest.Create(label, 1, rec_pars_dict)
            self.attached_device_names.append(label)
        else:
            mm = nest.Create('multimeter', 1, rec_pars_dict)
        self.attached_devices.append(mm)
        if ids is None:
            nest.Connect(mm, self.gids)
        else:
            nest.Connect(mm, list(ids))
        return mm

    @staticmethod
    def flush_records(device):
        """
        Delete all recorded events from extractors and clear device memory.

        :param device: gid of device from which to delete
        """
        nest.SetStatus(device, {'n_events': 0})
        if nest.GetStatus(device)[0]['to_file']:
            tools.utils.data_handling.remove_files(nest.GetStatus(device)[0]['filenames'])

    # TODO - Disconnect is only fully functional after nest 2.18
    def disconnect_devices(self, device_gid=None):
        """
        Disconnect recording devices connected to this population

        :param device_gid: gid of the device to disconnect (if None all devices are removed)
        """
        nest.Disconnect(self.attached_devices)

    def extract_activity(self, t_start=None, t_stop=None):
        """
        Extract activity recorded with all attached devices
        :return:
        """
        for dev in self.attached_devices:
            self.load_activity(dev, t_start=t_start, t_stop=t_stop)

    def load_activity(self, initializer, time_shift=0., t_start=None, t_stop=None):
        """
        Extract recorded activity from attached devices, convert it to SpikeList or AnalogList
        objects and store them appropriately.

        :param initializer: can be a string, or list of strings containing the relevant filenames where the
        raw data was recorded or be a gID for the recording device, if the data is still in memory
        :param time_shift: [float] usually the simulation resolution, this shift (to the left) on the time axis
                           is needed to align the population activity (spikes) with the sampled times, for which
                           the origin is t0 = stim_onset + encoder_delay
        :param t_start:
        :param t_stop:
        :return:
        """
        # TODO: save option!
        # if object is a string, it must be a file name; if it is a list of strings, it must be a list of filenames
        if isinstance(initializer, str) or isinstance(initializer, list):
            data = tools.utils.data_handling.extract_data_fromfile(initializer)
            if data is not None:
                if len(data.shape) != 2:
                    data = np.reshape(data, (int(len(data) / 2), 2))
                if data.shape[1] == 2:
                    spk_times = data[:, 1] - time_shift
                    neuron_ids = data[:, 0]
                    tmp = [(neuron_ids[n], spk_times[n]) for n in range(len(spk_times))]
                    self.spiking_activity = signals.spikes.SpikeList(tmp, np.unique(neuron_ids).tolist())
                    self.spiking_activity.complete(self.gids)
                else:
                    neuron_ids = data[:, 0]
                    times = data[:, 1]
                    if t_start is not None and t_stop is not None:
                        idx1 = np.where(times >= t_start)[0]
                        idx2 = np.where(times <= t_stop)[0]
                        idxx = np.intersect1d(idx1, idx2)
                        times = times[idxx] - time_shift
                        neuron_ids = neuron_ids[idxx]
                        data = data[idxx, :]
                    for nn in range(data.shape[1]):
                        if nn > 1:
                            sigs = data[:, nn]
                            tmp = [(neuron_ids[n], sigs[n]) for n in range(len(neuron_ids))]
                            self.analog_activity.append(
                                signals.analog.AnalogSignalList(tmp, np.unique(neuron_ids).tolist(),
                                                                times=times, t_start=t_start,
                                                                t_stop=t_stop))

        elif isinstance(initializer, tuple) or isinstance(initializer, int):
            if isinstance(initializer, int):
                status = nest.GetStatus([initializer])[0]['events']
            else:
                status = nest.GetStatus(initializer)[0]['events']

            # use length of status to determine which type signal was recorded
            # spike recorder
            if len(status) == 2:
                spk_times = status['times'] - time_shift
                neuron_ids = status['senders']
                tmp = [(neuron_ids[n], spk_times[n]) for n in range(len(spk_times))]
                if t_start is None and t_stop is None:
                    self.spiking_activity = signals.spikes.SpikeList(tmp, np.unique(neuron_ids).tolist())
                    self.spiking_activity.complete(self.gids)
                else:
                    self.spiking_activity = signals.spikes.SpikeList(tmp, np.unique(neuron_ids).tolist(), t_start=t_start,
                                                                     t_stop=t_stop)
                    self.spiking_activity.complete(self.gids)
            # analog signal recorder
            elif len(status) > 2:
                times = status['times']
                neuron_ids = status['senders']
                idxs = np.argsort(times)
                times = times[idxs] - time_shift
                neuron_ids = neuron_ids[idxs]
                if t_start is not None and t_stop is not None:
                    idxx = np.intersect1d(np.where(times >= t_start)[0], np.where(times <= t_stop)[0])
                    times = times[idxx]
                    neuron_ids = neuron_ids[idxx]
                rem_keys = ['times', 'senders']
                new_dict = {k: v[idxs] for k, v in status.items() if k not in rem_keys}
                self.analog_activity = []
                for k, v in new_dict.items():
                    tmp = [(neuron_ids[n], v[n]) for n in range(len(neuron_ids))]
                    self.analog_activity.append(signals.analog.AnalogSignalList(tmp, np.unique(neuron_ids).tolist(),
                                                                                times=times, t_start=t_start, t_stop=t_stop))
                    self.analog_activity_names.append(k)
        else:
            logger.error("Incorrect initializer...")

    def get_analog_activity_by_name(self, name):
        """

        :param name:
        :return:
        """
        if name not in self.analog_activity_names:
            raise ValueError('No analog activity named {}'.format(name))
        idx = self.analog_activity_names.index(name)
        return self.analog_activity[idx]


########################################################################################################################
class Network(ABC):
    @abstractmethod
    def initialize_states(self):
        pass

    @abstractmethod
    def report(self):
        """
        Print a description of the system
        """
        pass

    @abstractmethod
    def train(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, **kwargs):
        pass


def merge_population_activity(merged_populations, start=0., stop=1000., in_place=True):
    """
    Merge spike and analog data from the different populations, according to the already merged
    and registered populations. NO global merging here!

    :param merged_populations:
    :param populations:
    :param start: start time of merging
    :param stop: end time of merging (included?)
    :param save: [deprecated] not used anymore, only for backwards compatibility purposes
    :return:
    """
    merged_activity = {}
    if not empty(merged_populations):
        for m_pop in merged_populations.values():
            assert m_pop.is_merged

            m_spiking_activity = signals.spikes.SpikeList([], [], start, stop, m_pop.size)
            m_analog_activity = []

            # iterate over each parent population (ids) in the merged one and collect spiking and analog activities
            for child_seq_id, child_pop in enumerate(m_pop.children):
                # only add spiking activity if it's not empty
                if not child_pop.spiking_activity.empty():
                    child_spiking_activity = child_pop.spiking_activity
                    # truncate to correct start and end times if necessary
                    if not m_spiking_activity.time_parameters() == child_spiking_activity.time_parameters():
                        child_spiking_activity = child_spiking_activity.time_slice(start, stop)
                    m_spiking_activity.concatenate(child_spiking_activity)

                # add analog activity
                m_analog_activity.append(child_pop.analog_activity)
            if in_place:
                m_pop.spiking_activity = m_spiking_activity
                m_pop.analog_activity = m_analog_activity
            else:
                merged_activity.update({m_pop.name: {'spiking_activity': m_spiking_activity,
                                                     'analog_activity': m_analog_activity}})
    return merged_activity

