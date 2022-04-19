import copy
import itertools
import sys
import importlib
import numpy as np
from matplotlib import pyplot as plt

nest = None if importlib.util.find_spec("nest") is None else importlib.import_module("nest")

from .state_matrix import StateMatrix
from fna.tools import utils
from fna.tools.parameters import ParameterSet, validate_keys, extract_nestvalid_dict

logger = utils.logger.get_logger(__name__)


def set_recording_device(start=0., stop=sys.float_info.max, resolution=0.1, record_to='memory',
                         device_type='spike_detector', label=''):
    """
    Standard device parameters
    :param start: [float] device on time
    :param stop: [float] device off time
    :param resolution: [float] recording resolution
    :param record_to: [str] 'memory' or 'file'
    :param device_type: [str] 'spike_detector', 'multimeter', etc
    :param label: [str] device name
    :return: recording device parameter set
    """
    rec_devices = {
        'start': start,
        'stop': stop,
        'origin': 0.,
        'interval': resolution,
        'record_to': [record_to],
        'label': label,
        'model': device_type,
        'close_after_simulate': False,
        'flush_after_simulate': False,
        'flush_records': False,
        'close_on_reset': True,
        'withtime': True,
        'withgid': True,
        'withweight': False,
        'time_in_steps': False,
        'scientific': False,
        'precision': 3,
        'binary': False,
    }
    return ParameterSet(rec_devices)


class Extractor(object):
    """
    Small helper class to allow easy member variable access.
    # TODO remove lists where unnecessary, they were there to support sampling at different specified times
    """
    def __init__(self, label):
        self.label = label
        self.devices = []
        self.params = None
        self.population = None
        self.encoder = None
        self.resolution = None
        self.initial_states = None
        self.delay = None
        # for rate based sampling, activity will be a list of numpy matrices:
        #   - no averaging: one numpy array for each extracted batch, i.e., one final extraction = 1 array
        #   - averaging: 1 array for each averaged period
        # for fixed sampling, there's one numpy array for each sampling offset
        self.activity = []
        self.carry_activity = None
        # sampled times always refers to the network's time, i.e., it includes the stimulus onset + encoding delay,
        # but not the decoding delays
        self.sampled_times = []
        self.carry_sampled_times = None

    def get_populations_gids(self):
        return list(itertools.chain(*[pop.gids for pop in self.population]))

    def get_populations_names(self):
        return list(itertools.chain(*[pop.name for pop in self.population]))

    def has_activity(self):
        return any([len(x) for x in self.activity])


# ######################################################################################################################
class SpikingExtractor(object):
    """
    The Extractor reads population activity in response to patterned inputs,
    and extracts the network_architect state (according to specifications).
    """

    def __init__(self, initializer, variable_signal=False, stim_duration=None, stim_isi=None,
                 encoder_delay=0., input_resolution=0.1, to_memory=True):
        """

        :param initializer: ParameterSet object or dictionary specifying decoding parameters
        :param variable_signal:
        :param to_memory:
        :return
        """
        if isinstance(initializer, dict):
            initializer = ParameterSet(initializer)
        assert isinstance(initializer, ParameterSet), "StateExtractor must be initialized with ParameterSet or " \
                                                          "dictionary"

        self.stim_duration = stim_duration
        self.stim_isi = stim_isi
        self.encoder_delay = encoder_delay
        self.input_resolution = input_resolution
        self.variable_signal = variable_signal
        self.to_memory = to_memory
        self.sim_resolution = nest.GetKernelStatus('resolution')

        # variables initialized later during connection
        self.max_extractor_delay = 0.  # maximum extractor delay
        self.global_extraction = False  # will be set true when all activity is extracted

        self.extractor_pars = self._parse_extractor_parameters(initializer, to_memory)

        self.pop_extractors = {}  # population extractors
        self.enc_extractors = {}  # encoder extractors

    def _parse_extractor_parameters(self, extractor_pars_, to_memory=True):
        """
        Generates the extractor parameters through a combination of default values (see `_get_spikes_parameters` and
        `_get_vm_parameters` methods) and the values given by the user through the `extractor_pars_` argument.

        :param extractor_pars_:
        :param to_memory:
        :return:
        """
        extractor_pars = ParameterSet(extractor_pars_)

        for label, params in extractor_pars.items():
            validate_keys(required_keys=['variable'], any_keys=['population', 'encoder'], dictionary=params)

            if params.variable == 'spikes':
                new_pars = self._get_spikes_parameters(params)
            elif params.variable == 'V_m':
                new_pars = self._get_vm_parameters(to_memory, params)
            else:
                raise ValueError('Unknown state variable: {}'.format(params.variable))

            # by default, we do not store intermediate state matrices during training, only the test set
            if 'save' not in new_pars:
                new_pars['save'] = False

            # update corresponding extractor parameters for later return
            extractor_pars[label] = new_pars

        return ParameterSet(extractor_pars)

    @staticmethod
    def _parameter_check_sampling(pars):
        """
        Assertion checks for the sampling parameters.
        `average_states` + `sampling_rate` can not be using in combination with sampling_times, it's either or.
        """
        if 'average_states' in pars and pars.average_states:
            assert 'sampling_rate' in pars, "average_states=True requires the sampling_rate parameter to be set!"

        # sampling_rate or sampling_times has to be present
        if 'sampling_rate' in pars:
            assert 'sampling_times' not in pars, "sampling_times and sampling_rate can not be defined at the " \
                                                 "same time!"
        elif 'sampling_times' in pars:
            assert ('average_states' not in pars or pars.average_states is False) and 'sampling_rate' not in pars, \
                "average_states and sampling_rate can not be using in combination with sampling_times!"
            assert isinstance(pars.sampling_times, list), "sampling_times parameter must be of type list!"
        else:
            raise AttributeError('sampling_rate or sampling_times is required for the state extractor!')

    def _get_spikes_parameters(self, spike_pars):
        """
        Returns the parameter set dictionary for the case when the state variable is the low-pass filtered spike trains.
        Creates a default parameter dictionary but integrates any arguments provided by the user.

        :param spike_pars:
        :return:
        """
        self._parameter_check_sampling(spike_pars)

        d = {
            'state_specs': {
                'tau_m': 20. if 'filter_time' not in spike_pars else spike_pars.filter_time,
                'interval': 0.1  # this will later be changed according to the sampling strategy!
            },
            'reset_states': False,
            'average_states': False,
            'sample_isi': False,
            'standardize': False,
            'sampling_rate': None,
            'sampling_times': None,
        }
        d.update(spike_pars)
        return d

    def _get_vm_parameters(self, to_memory, vm_pars):
        """
        Returns the parameter set dictionary for the case when the state variable is the membrane potential.
        Creates a default parameter dictionary but integrates any arguments provided by the user.

        :param vm_pars:
        :return:
        """
        self._parameter_check_sampling(vm_pars)

        # will be changed according to the sampling strategy!
        rec_device = set_recording_device(start=0., record_to='memory' if to_memory else 'file', resolution=0.1)

        d = {
            'state_specs': utils.operations.copy_dict(rec_device, {'model': 'multimeter',
                                                                   'record_n': None,
                                                                   'record_from': ['V_m']}),
            'reset_states': False,
            'average_states': False,
            'sample_isi': False,
            'standardize': False,
            'sampling_rate': None,
            'sampling_times': None
        }
        d.update(vm_pars)

        if 'filter_time' in vm_pars:
            logger.warning('`filter_time` parameter will be ignored for analog extraction!')

        return d

    def _round_to_sim_resolution(self, number):
        """
        Rounds a number to the decimal digits of the simulation resultion.
        :param number:
        :return:
        """
        sim_res_digits = utils.operations.determine_decimal_digits(self.sim_resolution)
        return np.round(number, sim_res_digits)

    def _determine_extractor_delays(self, extractor):
        """
        Determine the connection delays involved in the different state extractors.
        This function sets the 'total_delay' dictionary entry for each state extractor, assuming there is
        a single such value in each case.

        :return: total_delay
        """
        # TODO NEST error (invalid connections returned by GetConnections) when using parrots. Temporary workaround
        # # get device status from NEST
        # status_dict = nest.GetStatus(nest.GetConnections(source=[extractor.devices[0]]))
        # tget_gids = [n['target'] for n in status_dict]
        #
        # device_delays = [d['delay'] for d in status_dict]
        # # extract unique delays, there should be exactly one for each state extractor!
        # unique_delays = np.unique(device_delays)
        # assert (len(unique_delays) == 1), "Heterogeneous delays in the state extractor (devices) are not supported.."
        # total_delay = float(unique_delays)
        #
        # # extract delay for decoding population of neurons (spikes)
        # pop_gids = extractor.get_populations_gids()
        # src_gids = [x for x in tget_gids if x in pop_gids]
        # if utils.operations.empty(src_gids):
        #     assert extractor.params.variable == 'spikes', \
        #         "No connections to {0} extractor".format(extractor.params.variable)
        #     assert np.array([nest.GetStatus([x])[0]['model'] == 'iaf_psc_delta' for x in tget_gids]).all(), \
        #         "No connections to {0} extractor".format(extractor.params.variable)
        #
        #     # get delay connection matrix from NESTConnector
        #     delay_matrix = net_arch.connectivity.NESTConnector.extract_connection_matrix(
        #         src_gids=pop_gids[:10], tgets_gids=tget_gids, key='delay', progress=False)
        #     # extract unique delays, there should be exactly one for each state extractor!
        #     unique_delays = np.unique(np.array(delay_matrix[delay_matrix.nonzero()].todense()))
        #     assert (len(unique_delays) == 1), "Heterogeneous delays in the state extractor are not supported.."
        #     total_delay += float(unique_delays)

        # # it's complicated. The NEST multimeter effectively has a sim_res delay in the case of V_m sampling,
        # # irrespective of the user-set delay in the NEST connection. We therefore must compensate this
        # # here as we do not consider it as an extraction delay, only the sim_res delay for `spikes` sampling.
        # total_delay -= self.sim_resolution
        total_delay = 0.1 if extractor.params.variable == 'V_m' else 0.2

        return total_delay

    def get_extractor(self, label):
        """

        :param label:
        :return:
        """
        # TODO add encoder support
        try:
            return self.pop_extractors[label]
        except KeyError:
            try:
                return self.enc_extractors[label]
            except KeyError:
                raise KeyError('Extractor [{}] not found!'.format(label))

    def get_labels(self):
        """
        Returns the labels of all state extractors.

        :return: [list] containing labels of every state extractor
        """
        return self.pop_extractors.keys()

    # TODO would this function make more sense in the Network class?
    def get_state_matrix(self, extractor_label):
        """

        :param extractor_label: [str]
        :return: [StateMatrix] state matrix object corresponding to `extractor_label`, or None, if it not available
        """
        population = self.get_extractor(extractor_label).population[0]
        try:
            return population.state_matrices[extractor_label]
        except KeyError:
            return None

    def _get_sampling_times(self, params, input_resolution, stim_onset):
        """
        Called when connecting the state extractor to the population, this function returns the sampling interval
        and the sampling offsets. The offsets are computed considering all delays in the system that might
        affect precise activity extraction.

        :param params:
        :param input_resolution:
        :param stim_onset:
        :return: [dict] {'interval': interval, 'offsets': offsets}
        """
        offsets = []
        sampling_specs = {}
        multimeter_delay = 0.1

        if self.stim_isi is None:
            self.stim_isi = 0.

        def __add_offset(offset_, samp_interval_, text_):
            # decoding the filtered spiketrains implies an extra delay of 0.1 due to the delta neurons
            if params.variable == 'spikes':
                offset_ += 0.1

            # offset_ = self._round_to_sim_resolution(offset_)
            offsets.append(offset_)
            logger.info("\t  - Extractor ({})".format(text_))
            logger.info("\t\t- offset = {} ms (stimulus onset + total delay + sim_res step)".format(str(offset_)))
            logger.info("\t\t- interval = {} ms".format(str(samp_interval_)))

        # -----------------------------------------------------------------------------------------------------------
        # this is a particular case where either the stimulus duration or the interval, or both have variable length.
        # The solution is a rate-based sampling with stimulus resolution
        if self.variable_signal:
            min_samp_rate = 1000. / input_resolution
            if self._is_rate_based_sampling(params):
                assert min_samp_rate <= params.sampling_rate, \
                    "For variable signal lengths, the sampling rate must be >= input resolution!"
                samp_interval = min(1000. / params.sampling_rate, min_samp_rate)
            else:
                # validate user-defined sampling times / offsets
                # TODO we should ensure that offsets < stim_duration + isi, for each stimulus. how?
                for samp_idx, samp_t in enumerate(params.sampling_times):
                    if samp_t != 'stim_offset' and \
                            utils.operations.determine_decimal_digits(input_resolution) < \
                            utils.operations.determine_decimal_digits(samp_t):
                        raise ValueError('Input resolution is lower than specified sampling offset!')
                samp_interval = input_resolution

            offset = stim_onset + self.encoder_delay + multimeter_delay
            __add_offset(offset, samp_interval, "variable {} sampling".format(
                'rate' if self._is_rate_based_sampling(params) else 'fixed offset'))
        # --------------------
        # rate-based sampling
        elif self._is_rate_based_sampling(params):
            samp_interval = 1000. / params.sampling_rate
            assert samp_interval <= (self.stim_duration + self.stim_isi), "Sampling rate must be >= stimulus duration + isi"
            assert np.isclose(self.stim_isi / samp_interval,
                              int(self.stim_isi / samp_interval)), "Incompatible sampling rate/isi!"
            assert np.isclose(self.stim_duration / samp_interval, int(self.stim_duration / samp_interval)), \
                "Sampling rate and stimulus duration not compatible!"

            offset = stim_onset + samp_interval - self.sim_resolution
            offset += self.encoder_delay + multimeter_delay

            __add_offset(offset, samp_interval, "rate sampling")
        # --------------------
        # sample at offset
        else:
            samp_interval = self.stim_duration + self.stim_isi
            assert params.sampling_times[0] == 'stim_offset' and len(params.sampling_times) == 1
            offset = self.stim_duration + stim_onset - self.sim_resolution  # last timepoint of input device
            offset += self.encoder_delay + multimeter_delay  # delays
            __add_offset(offset, samp_interval, "offset sampling")

        sampling_specs['interval'] = samp_interval
        sampling_specs['offset'] = offsets
        return sampling_specs

    def _attach_analog_recorder(self, extractor, target_gids):
        """
        Create and attach analog V_m recording devices to a population.

        :param extractor: [Extractor] state extractor object
        :param target_gids: NEST gids of objects to which the record is connected
        :return:
        """
        params = extractor.params
        offset = params.state_specs.offset[0]
        interval = params.state_specs.interval

        if self._is_rate_based_sampling(params):
            start = offset - self.sim_resolution
        else:
            start = 0.

        # this is a dirty hack to avoid a NEST multimeter error, present up to v2.20.0
        # TODO - this was causing an error in NEST 2.20.0
        offset_correction = int(nest.GetKernelStatus('time') / interval) * interval
        if offset >= offset_correction:
            offset -= offset_correction

        mm = nest.Create('multimeter', 1, {'record_from': ['V_m'],
                                           'record_to': ['memory'] if self.to_memory else 'file',
                                           'interval': interval,
                                           'start':  start,
                                           'offset': offset})
        extractor.devices = [mm[0]]
        nest.Connect(extractor.devices, target_gids, syn_spec={'delay': 0.1})

        # need to store some extractor data
        original_neuron_status = nest.GetStatus(target_gids)
        extractor.initial_states = np.array([x['V_m'] for x in original_neuron_status])

    def _attach_spike_recorder(self, extractor, target_gids):
        """
        Create and attach spike recorder devices to a population.

        :param extractor: [Extractor] state extractor object
        :param target_gids: NEST gids of objects to which the record is connected
        :return:
        """
        params = extractor.params
        offset = params.state_specs.offset[0]
        interval = params.state_specs.interval

        # create iaf_psc_delta neurons with delta synapses to compute the low-pass filtered spiking activity
        rec_neuron_pars = {'model': 'iaf_psc_delta', 'V_m': 0., 'E_L': 0., 'C_m': 1.,
                           'tau_m': params.state_specs['tau_m'],
                           'V_th': sys.float_info.max, 'V_reset': 0.,
                           'refractory_input': True, 'V_min': 0.}
        filter_neuron_specs = extract_nestvalid_dict(rec_neuron_pars, param_type='neuron')
        rec_neurons = nest.Create(rec_neuron_pars['model'], len(target_gids), filter_neuron_specs)

        if self._is_rate_based_sampling(params):
            start = offset - self.sim_resolution
        else:
            start = 0.
        # this is a dirty hack to avoid a NEST multimeter error, present up to v2.20.0, when sampling @offset
        offset_correction = int(nest.GetKernelStatus('time') / interval) * interval
        if offset >= offset_correction:
            offset -= offset_correction

        rec_mm = nest.Create('multimeter', 1, {'record_from': ['V_m'],
                                               'record_to': ['memory'] if self.to_memory else 'file',
                                               'interval': interval,
                                               'start':  start,
                                               'offset': offset})
        extractor.devices = [rec_mm[0]]

        nest.Connect(extractor.devices, rec_neurons, syn_spec={'delay': 0.1})
        # connect population to recording neurons with fixed delay == 0.1 (was rec_neuron_pars['interval'])
        nest.Connect(target_gids, rec_neurons, 'one_to_one',
                     syn_spec={'weight': 1., 'delay': 0.1, 'model': 'static_synapse'})

        extractor.initial_states = np.zeros((len(rec_neurons),))

    # TODO implement
    @staticmethod
    def _attach_other_recorder(extractor, target_gids):
        """
        Create and attach a recorder device (neither analog Vm nor spike recorder) to a population.

        :param extractor: [Extractor] state extractor object
        :param target_gids: NEST gids of objects to which the record is connected
        :return:
        """
        assert False
        params = extractor.params
        mm_specs = par.extract_nestvalid_dict(params.state_specs, param_type='device')

        for offset in params.state_specs.offsets:
            rec_mm = nest.Create('multimeter', 1, utils.operations.copy_dict(mm_specs, {'offset': offset}))
            extractor.devices.append(rec_mm[0])

        nest.Connect(extractor.devices, target_gids)
        extractor.initial_states = np.zeros((len(target_gids),))

    def _attach_devices(self, extractor, stim_onset):
        """
        Attach all devices to an extractor.
        :param extractor:
        :return:
        """
        # update extractor parameters with sampling times and corrected offsets
        sampling_specs = self._get_sampling_times(extractor.params, self.input_resolution, stim_onset)
        extractor.params.state_specs.update(sampling_specs)

        # gather all population gids that were gathered for this extractor
        populations_gids = extractor.get_populations_gids()

        self.pop_extractors[extractor.label] = extractor

        # attach the various recorders
        if extractor.params.variable == 'V_m':
            model = nest.GetStatus([populations_gids[0]])[0]['model']
            if model == 'poisson_neuron' or model == 'poisson-input-parrots':
                raise AttributeError("Can't attach analog recorder [{}] to parrot neurons!".format(extractor.label))
            self._attach_analog_recorder(extractor, populations_gids)
        elif extractor.params.variable == 'spikes':
            self._attach_spike_recorder(extractor, populations_gids)
        elif extractor.params.variable in nest.GetStatus(populations_gids[0])[0]['recordables']:
            self._attach_other_recorder(extractor, populations_gids)
        else:
            raise NotImplementedError("Acquisition from variable {0} not implemented".format(extractor.params.variable))

    def connect_populations(self, net, populations, stim_onset):
        """
        Connects all state extractors to the corresponding populations. Raises a warning if some populations are
        not found and ignores corresponding extractor entries.
        Called from the Network class.

        :param net: [Network] network object
        :param populations: [list] list of Population objects present in the network
        :param stim_onset:
        :return:
        """
        logger.info('Connecting extractors (SpikingExtractor):')

        for label, params in self.extractor_pars.items():
            if 'population' not in params:
                continue

            logger.info("\t- State [{}] from populations {} [{}]".format(label, params.population, params.variable))
            # find correct population(s) for this extractor
            extractor_populations = []
            if isinstance(params.population, str):
                try:
                    extractor_populations.append(populations[params.population])
                except KeyError:
                    logger.warn('No population {} defined in state extractor {}'.format(params.population, label))
                    continue
            elif isinstance(params.population, list):
                # this will be a merged population
                merged_pop_name, merged_pop = net.get_merged_population(params.population, store=True)
                # update the user defined list of populations with the merged population name
                params.population = merged_pop_name
                extractor_populations.append(merged_pop)
            else:
                raise ValueError('Invalid extractor parameters! Population name must be string!')

            # just ignore this extractor if no corresponding populations were found
            if len(extractor_populations) == 0:
                logger.warn('None of the populations defined in state extractor {} was found!'.format(label))
                continue

            extractor = Extractor(label)
            extractor.population = extractor_populations
            extractor.params = copy.deepcopy(params)

            self._attach_devices(extractor, stim_onset)

            extractor.delay = self._determine_extractor_delays(extractor)
            self.max_extractor_delay = max(self.max_extractor_delay, extractor.delay)

            logger.info("\t\t- total delays = {} ms ({} encoder + {} extractor delay)".format(
                extractor.delay + self.encoder_delay, self.encoder_delay, extractor.delay))
            logger.info("\t\t- NEST device id(s): {}".format(self.pop_extractors[label].devices))

    def reinitialize_extractors(self, stim_onset):
        """
        Creates new recording devices for all extractors. Required when there's a discontinuity
        in the stimulus onset times.

        :return:
        """
        for label, extractor in self.pop_extractors.items():
            logger.info("Reinitializing extractor [{}]".format(label))
            # stop recording of previous devices to save memory
            for dev in extractor.devices:
                logger.info("  - Stopping recording from device [{}]".format(dev))
                nest.SetStatus([dev], {'stop': stim_onset})
            # attach new devices
            self._attach_devices(extractor, stim_onset)

    # TODO implement
    def connect_encoders(self, encoders):
        """
        Connects all state extractors to the corresponding encoders. Raises a warning if some encoders are
        not found and ignores corresponding extractor entries.
        Called from the Network class.

        :param encoders: [dict] Dictionary with the encoder objects from the Network class
        :return:
        """
        return

    @staticmethod
    def _is_rate_based_sampling(params):
        return 'sampling_rate' in params and params.sampling_rate is not None

    @staticmethod
    def _is_fixed_offset_sampling(params):
        return 'sampling_times' in params

    def flush_records(self, extractor=None):
        """
        Clear data from NEST devices from one or all extractors. If extractor is not given, the devices are flushed
        for all extractors, otherwise only for the given one.

        :return:
        """
        extractors = [extractor] if extractor else self.pop_extractors.values()

        for ext in extractors:
            for dev in ext.devices:
                nest.SetStatus([dev], {'n_events': 0})
                if nest.GetStatus([dev])[0]['to_file']:
                    utils.data_handling.remove_files(nest.GetStatus([dev])[0]['filenames'])

            logger.info("  - Flushing devices of state extractor {0} {1} from populations {2}".format(
                ext.label, ext.devices, ext.params.population))

    def flush_states(self, carry=False):
        """
        Clear all extractor data (activity and sampled times).
        :param carry: whether to flush the carry arrays (activity from previous iterations)
        :return:
        """
        logger.info("Deleting activity data from all state extractors")
        for ext in self.pop_extractors.values():
            ext.activity = []
            ext.sampled_times = []

            if carry:
                ext.carry_activity = None
                ext.carry_sampled_times = None

    # TODO this needs to be checked!
    def reset_states(self, extractor=None):
        """
        Reset state extractor devices. If extractor is not given, the state is set to 0 for all extractors,
        otherwise only for the given one.

        :params extractor: [None or Extractor]
        :return:
        """
        assert False
        extractors = extractor if extractor else self.pop_extractors.values()

        for ext in extractors:
            params = ext.params
            # reset only if required
            if params.reset_states:
                population_gids = list(itertools.chain([]))
                if params.variable == 'V_m':
                    logger.info("Resetting V_m can lead to incorrect results!")

                    for idx, neuron_id in enumerate(ext.population.gids):
                        nest.SetStatus([neuron_id], {'V_m': ext.initial_states[idx]})
                elif params.variable == 'spikes':
                    recording_neuron_gids = nest.GetStatus(nest.GetConnections(ext.device), 'target')
                    for idx, neuron_id in enumerate(recording_neuron_gids):
                        nest.SetStatus([neuron_id], {'V_m': ext.initial_states[idx]})
                else:
                    try:
                        for idx, neuron_id in enumerate(ext.population.gids):
                            nest.SetStatus([neuron_id], {params.variable: ext.initial_states[idx]})
                    except ValueError:
                        logger.info("State variable {0} cannot be reset".format(params.variable))

    def _trim_and_store_overflow_samples(self, extractor, raw_responses, times, stim_info):
        """
        Removes the last sample from the raw responses in case it's an extra sample that doesn't belong to the current
        stimulus. The discarded samples belong to the next batch, and are therefore stored in the carry-over variables
        of the extractors.

        For each stimulus / batch, the simulation time window includes the maximum decoding delay present
        in the system. In case there is a spike extractor, there will be one extra (invalid) sample recorded by
        the V_m extractors, which must be handled appropriately.

        This is a special case which occurs only when the sampling resolution == input resolution.

        :param extractor: [Extractor] extractor object
        :param raw_responses: [np.array] all the (raw) responses read out from the current extractor/recording devices
        :param times: [np.array] the times corresponding to the raw responses
                      NOTE! The times are adjusted for the decoding delays, i.e., they only include the encoding delays
        :return:
        """
        sampling_interval = extractor.params.state_specs.interval
        # if self._is_rate_based_sampling(extractor.params): #and sampling_interval <= self.input_resolution:
        last_valid_sample = stim_info[-1][1]  # it's at the end of last signal (stimulus)
        n_trim = len(np.where(times > last_valid_sample  + 0.0001)[0])
        if n_trim > 0:
            extractor.carry_activity = raw_responses[:, -n_trim:]
            extractor.carry_sampled_times = times[-n_trim:]
            return raw_responses[:, :-n_trim], times[:-n_trim]

        return raw_responses, times

    def _get_variable_offset_sampling(self, extractor, stim_info, raw_responses, times):
        """
        Read recorded activity from one particular extractor and store it.

        :return: [np.array, np.array] population responses sampled by the extractor and the according sampled times
        """
        params = extractor.params
        sampling_interval = params.state_specs.interval
        sampled_indices = []

        window_responses, times = self._trim_and_store_overflow_samples(extractor, raw_responses, times, stim_info)

        # select the required samples
        cur_idx = 0
        for stim in stim_info:
            t_start, t_stop, t_isi = stim
            n_samp_stim = int(round(t_stop - t_start - t_isi) / sampling_interval)     # samples during stimulus
            n_samp_isi = int(round(t_isi / sampling_interval))                         # samples during isi

            # there are #n_samp_stim samples belonging to the current stimulus, and their indices are
            # [cur_idx, cur_idx + 1, ... cur_idx + n_samp_stim - 1], hence the -1 for the sample @ stimulus offset
            sample_x = cur_idx + n_samp_stim - 1

            sampled_indices.append(sample_x)  # stored index in response array of the sample
            cur_idx += n_samp_stim + n_samp_isi  # move to next token

        sampled_responses = window_responses[:, sampled_indices]
        return sampled_responses, times[sampled_indices]

    def _get_offset_sampling(self, extractor, stim_info, raw_responses, times):
        """
        Read recorded activity from one particular extractor and store it.

        :return: [np.array, np.array] population responses sampled by the extractor and the according sampled times
        """
        window_responses, times = self._trim_and_store_overflow_samples(extractor, raw_responses, times, stim_info)
        return window_responses, times

    def _get_rate_sampling(self, extractor, stim_info, raw_responses, times):
        """
        Extracts the required state responses (vectors) from the raw device responses in the case when the stimulus
        duration and/or ISI have variable length.
        Note: for this variable scenario, the multimeter.offset is set to stim_onset + total delay, with the devices
              recording everything beginning at this offset and at the specified resolution. The first value
              in the raw responses corresponds to the value sampled at multimeter.offset.

        :return: [np.array] id x sampled values
        """
        params = extractor.params
        # arr_responses = raw_responses.as_array()
        window_responses, times = self._trim_and_store_overflow_samples(extractor, raw_responses, times, stim_info)

        if not params.average_states and params.sample_isi:
            return window_responses, times

        sampling_interval = params.state_specs.interval
        state_vectors = []
        sampled_times = []

        # select the required samples
        # discard the very first sample (start from 1), as it corresponds to the stimulus onset
        cur_idx = 0
        for stim in stim_info:
            t_start, t_stop, t_isi = stim
            n_samples_stim = int(round(t_stop - t_start - t_isi) / sampling_interval)  # samples during stimulus
            n_samples_isi = int(round(t_isi / sampling_interval))  # samples during isi

            # remove samples during isi
            if not params.sample_isi:
                start_idx = cur_idx  # when the stimulus starts
                stop_idx = start_idx + n_samples_stim  # where the stimulus ends and isi starts
            # don't remove anything, take full step
            elif params.sample_isi:
                start_idx = cur_idx  # when the stimulus starts
                stop_idx = start_idx + n_samples_stim + n_samples_isi # full step, where the isi ends
            else:
                raise NotImplementedError('Unknown value for sample_isi parameter!')

            state_vector_window = window_responses[:, start_idx:stop_idx]
            if params.average_states:
                # average across time window (columns), rows are still the neuron ids
                state_vectors.append(np.mean(state_vector_window, axis=1, keepdims=True))
                # the last sample included is at index stop_idx - 1, not stop_idx
                sampled_times.append(times[stop_idx - 1])
            else:
                state_vectors.append(state_vector_window)
                sampled_times.extend(times[start_idx:stop_idx])

            # move to next token
            cur_idx += n_samples_stim + n_samples_isi

        # concatenate the different averaged state vectors if extraction period spans multiple stimulus steps
        # shape will be N x (averaged) valid samples
        return np.concatenate(state_vectors, axis=1), sampled_times

    @staticmethod
    def _get_raw_responses_from_device(dev, time_shift):
        """
        Returns a matrix with the responses extracted from device `dev`.

        :param dev:
        :param time_shift:
        :param start:
        :param stop:
        :return: [np.array] N x n_samples
        """
        status_dict = nest.GetStatus([dev])[0]['events']
        # adjust the extracted times for the extractor delay
        times = status_dict['times'] - time_shift

        unique_senders = sorted(np.unique(status_dict['senders']))
        responses = []

        for id_ in unique_senders:
            indices = np.where(status_dict['senders'] == id_)[0]
            assert np.array_equal(sorted(times[indices]), times[indices])
            responses.append(status_dict['V_m'][indices])
        responses = np.array(responses)

        # TODO remove this after a while, it's here just to ensure correctness
        # try:
        #     tmp_resp = [(status_dict['senders'][n], status_dict['V_m'][n]) for n in range(len(status_dict['senders']))]
        #     raw_responses = signals.analog.AnalogSignalList(tmp_resp, np.unique(status_dict['senders']).tolist(),
        #                                                     times=times, t_start=start, t_stop=stop)
        #     assert np.array_equal(responses, raw_responses.as_array())
        # except:
        #     pass

        return times, responses

    def extract_activity(self, extractor, stim_info=None, carry_end=False, carry_begin=False):
        """
        Read recorded activity from one particular extractor and store it in extractor.activity.
        For variable stimulus duration and/or ISI, the activity is sampled at input resolution and careful
        post-processing is required to account for all (encoding, extraction) delays.

        :param extractor: [Extractor] object
        :param stim_info: [list] list of tuples with each stimulus token's start, end and interval time
        :param carry_end:
        :param carry_begin: [bool] whether to append carry samples from previous iterations to the beginning of
                                   the current samples
        :return
        """
        logger.info("Extracting and storing recorded activity from state extractor {}".format(extractor.label))

        devices = extractor.devices
        variable = extractor.params.variable
        time_shift = extractor.delay

        if not nest.GetStatus(devices)[0]['to_memory']:
            raise NotImplementedError("Extract to file not yet supported")

        # read data in memory
        dev = extractor.devices[0]
        rec_times, rec_responses = self._get_raw_responses_from_device(dev, time_shift)
        unique_times = np.unique(rec_times)

        # store samples from previous batch to be added later, if needed
        carry_activity = extractor.carry_activity
        carry_sampled_times = extractor.carry_sampled_times

        if carry_begin and carry_activity is not None:
            if len(rec_responses) == 0:
                rec_responses = carry_activity
                unique_times = carry_sampled_times
            else:
                rec_responses = np.concatenate((carry_activity, rec_responses), axis=1)
                unique_times = np.concatenate((carry_sampled_times, unique_times))
            # reset carry arrays
            extractor.carry_activity = None
            extractor.carry_sampled_times = None

        # ################################################
        # rate sampling
        if self._is_rate_based_sampling(extractor.params):
            logger.info("  - Reading extractor {0} [{1}] with rate sampling at ".format(dev, variable))
            samp_responses, samp_times = self._get_rate_sampling(extractor, stim_info, rec_responses, unique_times)
        # ################################################
        # sample at offset
        else:
            logger.info("  - Reading extractor {0} [{1}] at sampling offset".format(dev, variable))
            if self.variable_signal:
                samp_responses, samp_times = self._get_variable_offset_sampling(extractor, stim_info,
                                                                                rec_responses, unique_times)
            else:
                samp_responses, samp_times = self._get_offset_sampling(extractor, stim_info,
                                                                       rec_responses, unique_times)

        # add samples from previous batch, if required
        if carry_end and not carry_begin and carry_activity is not None:
            extractor.activity = [carry_activity] + extractor.activity
            extractor.sampled_times = [carry_sampled_times] + extractor.sampled_times
            # samp_responses = np.concatenate((carry_activity, samp_responses), axis=1)
            # samp_times = np.concatenate((carry_sampled_times, samp_times))

        extractor.activity.append(samp_responses)
        extractor.sampled_times.append(samp_times)

        # flush outside the loop so that activity from all devices of the extractor can be stored
        self.flush_records(extractor)

    def extract_global_states(self, stim_info, batch_end=False, carry_begin=False):
        """
        Extract activity of all registered state extractors.

        :param stim_info:
        :param batch_end:
        :param carry_begin:
        :return:
        """
        for label, extractor in self.pop_extractors.items():
            self.extract_activity(extractor, stim_info, batch_end, carry_begin)

            if batch_end:
                extractor.activity = [np.concatenate(extractor.activity, axis=1)]
                extractor.sampled_times = [np.concatenate(extractor.sampled_times)]

        self.global_extraction = True

    def compile_state_matrices(self, dataset_label, force_save=False):
        """
        Compile the recorded activity into state matrices. This is done per batch / (test) dataset.

        :param dataset_label: batch or test dataset label, used for saving the data
        :param force_save: if True, overrides extractor `save` parameter. Used to ensure saving the test dataset
        :return:
        """
        logger.info('Compiling state matrices...')

        # ensure that all activity is extracted
        if not self.global_extraction:
            self.extract_global_states()

        for label, extractor in self.pop_extractors.items():
            assert extractor.has_activity(), "Activity of extractor {} should've been extracted before!".format(label)

            # store the state matrix in the population
            save = extractor.params['save'] or force_save
            sm = StateMatrix(extractor.activity[0], extractor.label, state_var=extractor.params.variable,
                             population=extractor.population[0].name, sampled_times=extractor.sampled_times,
                             dataset_label=dataset_label, save=save)
            if "standardize" in extractor.params.keys():
                if extractor.params['standardize']:
                    sm.standardize()
            extractor.population[0].state_matrices[label] = sm

    def report(self):
        """
        Prints some extraction stats.
        :return:
        """
        logger.info("")
        logger.info("==========================")
        logger.info("SNN StateExtractor summary")
        for label, extractor in self.pop_extractors.items():
            logger.info("---------------")
            logger.info("State extractor [{}] from population(s) {} with gid(s) {}".format(
                label, extractor.params.population, extractor.devices))

            ##################################
            if extractor.params.sampling_rate:
                logger.info("  - gathered {} samples via rate-based sampling from {}, extractor delay {} ms".format(
                    len(extractor.sampled_times[0]), extractor.params.variable, extractor.delay))

                if extractor.params.average_states:
                    msg_isi = "and ISI" if extractor.params.sample_isi else "only, not ISI"
                    logger.info("    - samples averaged across stimulus duration {}".format(msg_isi))
                    logger.info("    - note: only the last sampled timepoint from the averaged window is shown")
                logger.info("    - network time (incl. encoder delay):")
                logger.info("      {} ... [{}]".format(extractor.sampled_times[0][:3], extractor.sampled_times[0][-1]))
                logger.info("    - NEST kernel time (incl. all delays):")
                kernel_sampled_times = np.array(extractor.sampled_times) + extractor.delay
                logger.info("      {} ... [{}]".format(kernel_sampled_times[0][:3], kernel_sampled_times[0][-1]))
            ##################################
            # fixed sampling
            else:
                logger.info("  - gathered {} samples @offset, sampling from {}, extractor delay {} ms".format(
                    len(extractor.sampled_times[0]), extractor.params.variable, extractor.delay))

                logger.info("    - network time (incl. encoder delay):")
                logger.info("      {} ... [{}]".format(extractor.sampled_times[0][:3], extractor.sampled_times[0][-1]))
                logger.info("    - NEST kernel time (incl. all delays):")
                kernel_sampled_times = np.array(extractor.sampled_times) + extractor.delay
                logger.info("      {} ... [{}]".format(kernel_sampled_times[0][:3], kernel_sampled_times[0][-1]))
        logger.info("")

    def plot_samples(self, label, n_samples=50, device_idx=0, neuron_id=None, stim_info=None,
                     start_t=None, stop_t=None, save=None, marker='o'):
        """

        :param label:
        :param n_samples:
        :param neuron_id:
        :param stim_info:
        :param start_t:
        :param stop_t:
        :param save:
        :param marker:
        :return:
        """
        logger.info('===========================')
        logger.info('Example extractor activity: [{}]'.format(label))

        extractor = self.pop_extractors[label]
        net_times = np.array(extractor.sampled_times[0])
        raw_kernel_times = net_times + extractor.delay

        if start_t is None:
            start_t = net_times[0]
        if stop_t is None:
            stop_t = raw_kernel_times[n_samples - 1]

        population = extractor.population[0]
        if isinstance(population.spiking_activity, list) and len(population.spiking_activity) == 0:
            logger.info("No spikes recorded for population {}.".format(extractor.population[0].name))
            return

        if neuron_id is None:
            spikelist = population.spiking_activity.id_slice(1)
            neuron_id = spikelist.id_list[0]
        else:
            spikelist = population.spiking_activity.id_slice([neuron_id])
        min_neuron_id = min(population.spiking_activity.id_list)
        # shift spikelist so that current neuron has id=1
        spikelist.id_offset(-spikelist.id_list[0] + 1)

        spikelist = spikelist.time_slice(t_start=0., t_stop=max(raw_kernel_times))

        fig = plt.figure(figsize=(20, 4))
        ax = fig.add_subplot(1, 1, 1)
        if not utils.operations.empty(population.spiking_activity):
            window_indices = np.where((start_t <= net_times) & (net_times < stop_t))
            neuron_samples = extractor.activity[device_idx][neuron_id - min_neuron_id][window_indices]
            kernel_times = raw_kernel_times[window_indices]
            net_times = net_times[window_indices]

            # plot extracted activity, for both the adjusted and raw sampling times
            ax.plot(kernel_times, neuron_samples, marker, color='r', label='kernel time (+ all delays)')
            ax.plot(net_times, neuron_samples, marker, color='g', label='network time (encoder delay)')

            # plot a vertical line for each spike
            for idx, spk_time in enumerate(spikelist.spiketrains[1].spike_times):
                ax.axvline(x=spk_time, color='b', label='spike times' if idx == 0 else None)

            if stim_info:
                for t_start, t_stop, t_isi in stim_info:
                    # logger.info('{}, {}, {}'.format(t_start, t_stop, t_isi))
                    ax.axvspan(t_start, t_stop - t_isi, facecolor='g', alpha=0.25)
                    ax.axvspan(t_stop - t_isi, t_stop, facecolor='r', alpha=0.25)

            ax.set_xlim(xmin=start_t - 1, xmax=stop_t+1.)
            ax.set_ylim(bottom=-.2, top=max(1.1, max(neuron_samples + 1)))

            ax.set_xlabel('Time [ms]')
            ax.legend(loc='best')

            if save:
                fig.savefig(save)
            else:
                plt.show()
        else:
            print("Selected neuron did not spike, please run the code fragment again!")
