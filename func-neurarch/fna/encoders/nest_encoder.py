import itertools
import importlib

nest = None if importlib.util.find_spec("nest") is None else importlib.import_module("nest")

from fna.decoders.extractors import set_recording_device
from fna.tools.parameters import extract_nestvalid_dict
from fna.tools.parameters import ParameterSet
from fna.encoders.generators import Generator
from fna.tools.network_architect import Population
from fna.tools import utils

logger = utils.logger.get_logger(__name__)


class NESTEncoder(Generator):
    """
    Wrapper for devices used to convert inputs into spikes/currents/rates, using NEST generator
    devices.
    """
    def __init__(self, generator_device, stim_seq=None, label=None, dim=1, input_resolution=0.1, prng=None):
        """
        NEST encoding layer instance constructor.

        :param generator_device: dict
        :param stim_seq: stimulus sequence - SpikeList, AnalogSignalList or iterator
        :param dim: embedding dimensionality (number of unique devices to generate)
        :param input_resolution: dt
        :param prng: random number generator
        """
        self.total_delay = 0.
        self.n_generators = dim
        self._is_connected = False
        self.parrots = None
        logger.info("Creating Generators: ")
        super().__init__(generator_device, device_label=label, device_resolution=input_resolution,
                         input_signal=stim_seq, dims=dim)
        logger.info("- {0!s} [{1!s}-{2!s}]".format(self.name, min(self.gids), max(self.gids)))

    def add_parrots(self, N=None, dt=0.1):
        """
        Record the encoder activity with parrot neurons (if device in ['spike_generator', 'inh_poisson']
        :return:
        """
        logger.info("Creating input parrots for {0!s}:".format(self.name))
        valid_devices = ['spike_generator', 'inhomogeneous_poisson_generator']
        if self.model not in valid_devices:
            raise ValueError("Cannot add parrots to {0!s}".format(self.model))
        if N is None:
            N = self.input_dimension
        parrot_population_pars = {
            'pop_names': 'input-parrots',
            'n_neurons': N,
            'neuron_pars': {},
            'randomize': {},
            'topology': False,
            'topology_dict': None,
            'record_spikes': extract_nestvalid_dict(set_recording_device(device_type='spike_detector',
                                                                         resolution=dt), param_type='device'),
            'spike_device_pars': {},
            'record_analogs': False,
            'analog_device_pars': None}

        # create a normal population
        pop_dict = {k: v for k, v in parrot_population_pars.items()}

        # create neuron model named after the population
        nest.CopyModel('parrot_neuron', self.name+'-parrots')

        # create population
        gids = nest.Create(self.name+'-parrots', n=int(N))
        gids = sorted(gids)

        # set up population objects
        pop_dict.update({'gids': gids, 'is_subpop': False})
        self.parrots = Population(ParameterSet(pop_dict))
        logger.info("- Population {0!s}, with ids [{1!s}-{2!s}]".format(self.name+'-parrots', min(gids), max(gids)))
        self.parrots.record_spikes(parrot_population_pars['record_spikes'])
        synapse_name = str(self.name) + '-syn'
        if synapse_name not in nest.Models():
            nest.CopyModel('static_synapse', synapse_name)
        nest.Connect(list(itertools.chain(*self.gids)), self.parrots.gids, conn_spec={'rule': 'one_to_one'},
                     syn_spec={'model': synapse_name, 'delay': nest.GetKernelStatus('resolution')})

    def add_noise(self):
        """
        Add noise to the input generation process
        :return:
        """
        nest.Create('noise_generator') # TODO
