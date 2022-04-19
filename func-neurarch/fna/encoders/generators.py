"""
========================================================================================================================
Generators
========================================================================================================================
(this documentation is incomplete)

Classes:
--------
StochasticGenerator - Stochastic process generator
InputNoise          - generate and store AnalogSignal object referring to the noise to add to the input signal u(t)

Functions:
----------
set_stimulating_device()

========================================================================================================================
"""
import time
import types
import sys
import numpy as np
import importlib

nest = None if importlib.util.find_spec("nest") is None else importlib.import_module("nest")
# noinspection PyUnresolvedReferences

from fna import tools
from fna.tools import parameters
from fna.tools import signals
from fna.tools.utils import system
from fna.tools.signals import SpikeList

logger = tools.utils.logger.get_logger(__name__)


def set_stimulating_device(start=0., stop=sys.float_info.max, origin=0., device_type='spike_generator'):
    """
    Standard device parameters
    :param start: [float] device on time
    :param stop: [float] devide off time
    :param origin: [float]
    :param device_type: [str] 'spike_detector', 'multimeter', etc
    :return: recording device parameter set
    """
    stim_devices = {
        'allow_offgrid_spikes': True,
        'model': device_type,
        'origin': origin,
        'start': start,
        'stop': stop,
    }
    return parameters.ParameterSet(stim_devices)


class StochasticGenerator:
    """
    Stochastic process generator
    ============================
    (adapted from NeuroTools)

    Generate stochastic processes of various types and return them as SpikeTrain or AnalogSignal objects.

    Implemented types:
    ------------------
    a) Spiking Point Process - poisson_generator, inh_poisson_generator, gamma_generator, !!inh_gamma_generator!!,
    inh_adaptingmarkov_generator, inh_2Dadaptingmarkov_generator

    b) Continuous Time Process - OU_generator, GWN_generator, continuous_rv_generator (any other distribution)
    """

    def __init__(self, rng=None):
        """
        :param rng: random number generator state object (optional). Either None or a numpy.random.default_rng object,
        or an object with the same interface

        If rng is not None, the provided rng will be used to generate random numbers, otherwise StGen will create
        its own rng.
        """
        if rng is None:
            logger.warning("RNG for StochasticGenerator not set! Results may not be reproducible!")
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def poisson_generator(self, rate, t_start=0.0, t_stop=1000.0, array=False, debug=False):
        """
        Returns a SpikeTrain whose spikes are a realization of a Poisson process
        with the given rate (Hz) and stopping time t_stop (milliseconds).

        Note: t_start is always 0.0, thus all realizations are as if
        they spiked at t=0.0, though this spike is not included in the SpikeList.

        :param rate: the rate of the discharge (in Hz)
        :param t_start: the beginning of the SpikeTrain (in ms)
        :param t_stop: the end of the SpikeTrain (in ms)
        :param array: if True, a numpy array of sorted spikes is returned,
                      rather than a SpikeTrain object.

        :return spikes: SpikeTrain object

        Examples:
        --------
            >> gen.poisson_generator(50, 0, 1000)
            >> gen.poisson_generator(20, 5000, 10000, array=True)
        """

        n = (t_stop - t_start) / 1000.0 * rate
        number = np.ceil(n + 3 * np.sqrt(n))
        if number < 100:
            number = min(5 + np.ceil(2 * n), 100)

        if number > 0:
            isi = self.rng.exponential(1.0 / rate, int(number)) * 1000.0
            if number > 1:
                spikes = np.add.accumulate(isi)
            else:
                spikes = isi
        else:
            spikes = np.array([])

        spikes += t_start
        i = np.searchsorted(spikes, t_stop)

        extra_spikes = []
        if i == len(spikes):
            # ISI buf overrun
            t_last = spikes[-1] + self.rng.exponential(1.0 / rate, 1)[0] * 1000.0

            while t_last < t_stop:
                extra_spikes.append(t_last)
                t_last += self.rng.exponential(1.0 / rate, 1)[0] * 1000.0

            spikes = np.concatenate((spikes, extra_spikes))

            if debug:
                logger.debug("ISI buf overrun handled. len(spikes)=%d, len(extra_spikes)=%d" % (len(spikes),
                                                                                                len(extra_spikes)))

        else:
            spikes = np.resize(spikes, (i,))

        if not array:
            spikes = signals.spikes.SpikeTrain(spikes, t_start=t_start, t_stop=t_stop)

        if debug:
            return spikes, extra_spikes
        else:
            return spikes

    def gamma_generator(self, a, b, t_start=0.0, t_stop=1000.0, array=False, debug=False):
        """
        Returns a SpikeTrain whose spikes are a realization of a gamma process
        with the given shape a, b and stopping time t_stop (milliseconds).
        (average rate will be a*b)

        Note: t_start is always 0.0, thus all realizations are as if
        they spiked at t=0.0, though this spike is not included in the SpikeList.

        :param a,b: the parameters of the gamma process
        :param t_start: the beginning of the SpikeTrain (in ms)
        :param t_stop: the end of the SpikeTrain (in ms)
        :param array: if True, a numpy array of sorted spikes is returned, rather than a SpikeTrain object.

        Examples:
        --------
            >> gen.gamma_generator(10, 1/10., 0, 1000)
            >> gen.gamma_generator(20, 1/5., 5000, 10000, array=True)
        """
        n = (t_stop - t_start) / 1000.0 * (a * b)
        number = np.ceil(n + 3 * np.sqrt(n))
        if number < 100:
            number = min(5 + np.ceil(2 * n), 100)

        if number > 0:
            isi = self.rng.gamma(a, b, number) * 1000.0
            if number > 1:
                spikes = np.add.accumulate(isi)
            else:
                spikes = isi
        else:
            spikes = np.array([])

        spikes += t_start
        i = np.searchsorted(spikes, t_stop)

        extra_spikes = []
        if i == len(spikes):
            # ISI buf overrun
            t_last = spikes[-1] + self.rng.gamma(a, b, 1)[0] * 1000.0

            while t_last < t_stop:
                extra_spikes.append(t_last)
                t_last += self.rng.gamma(a, b, 1)[0] * 1000.0

            spikes = np.concatenate((spikes, extra_spikes))

            if debug:
                logger.debug("ISI buf overrun handled. len(spikes)=%d, len(extra_spikes)=%d" % (len(spikes),
                                                                                                len(extra_spikes)))
        else:
            spikes = np.resize(spikes, (i,))

        if not array:
            spikes = signals.spikes.SpikeTrain(spikes, t_start=t_start, t_stop=t_stop)

        if debug:
            return spikes, extra_spikes
        else:
            return spikes

    def OU_generator(self, dt, tau, sigma, y0, t_start=0.0, t_stop=1000.0, rectify=False, array=False, time_it=False):
        """
        Generates an Ornstein-Uhlenbeck process using the forward euler method. The function returns
        an AnalogSignal object.

        :param dt: the time resolution in milliseconds of th signal
        :param tau: the correlation time in milliseconds
        :param sigma: std dev of the process
        :param y0: initial value of the process, at t_start
        :param t_start: start time in milliseconds
        :param t_stop: end time in milliseconds
        :param array: if True, the functions returns the tuple (y,t)
                      where y and t are the OU signal and the time bins, respectively,
                      and are both numpy arrays.
        :return AnalogSignal
        """
        if time_it:
            t1 = time.time()

        t = np.arange(t_start, t_stop, dt)
        N = len(t)
        y = np.zeros(N, float)
        y[0] = y0
        fac = dt / tau
        gauss = fac * y0 + np.sqrt(2 * fac) * sigma * self.rng.standard_normal(N - 1)
        mfac = 1 - fac

        # python loop... bad+slow!
        for i in range(1, N):
            idx = i - 1
            y[i] = y[idx] * mfac + gauss[idx]

        if time_it:
            logger.info(time.time() - t1)
        if rectify:
            y[y < 0] = 0.

        if array:
            return (y, t)
        else:
            return signals.analog.AnalogSignal(y, dt, t_start, t_stop)

    # @staticmethod  # this can't be static if we want reproducibility
    def GWN_generator(self, amplitude=1., mean=0., std=1., t_start=0.0, t_stop=1000.0, dt=1.0, rectify=True,
                      array=False):
        """
        Generates a Gaussian White Noise process. The function returns an AnalogSignal object.

        :param amplitude: maximum amplitude of the noise signal
        """

        t = np.arange(t_start, t_stop, dt)
        wn = amplitude * self.rng.normal(loc=mean, scale=std, size=len(t))

        if rectify:
            wn[wn < 0] = 0.

        if array:
            return (wn, t)
        else:
            return signals.analog.AnalogSignal(wn, dt, t_start, t_stop)

    @staticmethod
    def continuous_rv_generator(function, amplitude=1., t_start=0.0, t_stop=1000.0, dt=1.0, rectify=True,
                                array=False, **kwargs):
        """
        Generates a realization of a continuous noise process by drawing iid values from the distribution specified by
        function and parameterized by **kwargs
        :param function: distribution function (e.g. np.random.poisson)
        Note: **kwargs must correspond to the function parameters
        """

        t = np.arange(t_start, t_stop, dt)
        if isinstance(function, str):
            function = eval(function)
        s = function(size=len(t), **kwargs)
        s *= amplitude

        if rectify:
            s[s < 0] = 0.
        if array:
            return s, t
        else:
            return signals.analog.AnalogSignal(s, dt, t_start, t_stop)


class Generator:
    """
    Generate input to the network_architect, generator is a NEST device!
    The generator always inherits the dimensionality of its input and contains the
    connectivity features to its target (N_{input}xN_{target})
    """

    def __init__(self, device_name, device_label, device_resolution=0.1, input_signal=None, dims=None):
        """
        Generator instance constructor
        :param device_name: [str] NEST model name
        :param device_label: [str] label for this device
        :param input_signal: [SpikeList or AnalogSignalList] if a continuous input signal is already available
        :param dims: [int] embedding dimensions (number of generators to create)
        """
        self.input_dimension = dims
        self.model = device_name
        self.gids = []
        if device_label is None:
            self.name = device_name
        else:
            self.name = device_label
        self.resolution = device_resolution

        device_params = set_stimulating_device(device_type=device_name)

        if self.name != self.model:
            nest.CopyModel(self.model, self.name)
        nest.SetDefaults(self.name, parameters.extract_nestvalid_dict(device_params, param_type='device'))

        for nn in range(self.input_dimension):
            self.gids.append(nest.Create(self.name))

        if input_signal is not None and not isinstance(input_signal, types.GeneratorType):
            self.update_state(input_signal)

    def update_state(self, signal, prev_signal_ids=None, ids_to_update=None):
        """
        For online generation, the input signal is given iteratively
        and the state of the NEST generator objects needs to be updated.
        :param signal: (SpikeList or AnalogSignalList) object containing
                       the values (e.g., spike times) to update the generators with
        :param prev_signal_ids: [list] ids of signals that were active in the previous timestep and need to be reset
        :param ids_to_update: [list] ids of signals that will be active in the current (next)
                              timestep and must be updated
        :return:
        """
        # for a SpikeList object, update the corresponding NEST spike generators
        if isinstance(signal, signals.spikes.SpikeList):
            rounding_precision = tools.utils.operations.determine_decimal_digits(self.resolution)
            for nn in signal.id_list:
                spike_times = [round(n, rounding_precision) for n in signal[nn].spike_times]  # to be sure
                nest.SetStatus(self.gids[nn], {'spike_times': spike_times})
        else:
            assert isinstance(signal, signals.analog.AnalogSignalList), "Incorrect signal format!"

            if self.input_dimension != len(signal):
                self.input_dimension = len(signal)

            # return here model doesn't requires update to save time
            if self.model != 'step_current_generator' and self.model != 'inhomogeneous_poisson_generator':
                return

            if ids_to_update is not None:
                assert isinstance(ids_to_update, list), "Signal ids to update in generators must be given in list!"
                assert isinstance(prev_signal_ids, list) or prev_signal_ids is None, \
                    "Signal ids to update in generators must be given in list!"

                # in some cases, the same signal id might be updated twice, but this is not an issue
                if prev_signal_ids is not None:
                    ids_to_update += prev_signal_ids

                for nn in ids_to_update:
                    t_axis = signal.time_axis()
                    s_data = signal[nn].raw_data()
                    if len(t_axis) != len(s_data):
                        t_axis = t_axis[:-1]

                    system.update_generator_states(model=self.model, gids=self.gids[nn],
                                                               signal_amplitudes=s_data,
                                                               signal_times=t_axis)
            else:
                for nn in range(self.input_dimension):
                    t_axis = signal.time_axis()
                    s_data = signal[nn].raw_data()
                    if len(t_axis) != len(s_data):
                        t_axis = t_axis[:-1]

                    system.update_generator_states(model=self.model, gids=self.gids[nn],
                                                               signal_amplitudes=s_data,
                                                               signal_times=t_axis)


def generate_spike_template(rate, duration, n_neurons=None, resolution=0.01, rng=None, store=False):
    """
    Generates a spatio-temporal spike template (an instance of `frozen noise')
    :param n_neurons: Number of neurons that compose the pattern
    :param rate: spike rate in the template or iterable of spike rates
    :param duration: [ms] total duration of template
    :param resolution:
    :param rng: random number generator state object (optional). Either None or a numpy.random.RandomState object,
        or an object with the same interface
    :param store: save the template in the provided path
    :return: SpikeList object
    """
    try:
        iter(rate)
        assert n_neurons is None or len(rate) == n_neurons, 'chose n_neurons to be equal to len(rate) or None'
    except TypeError:
        rate = np.ones(n_neurons) * rate

    gen = StochasticGenerator(rng=rng)
    times = []
    ids = []
    rounding_precision = tools.utils.operations.determine_decimal_digits(resolution)
    for n, r in enumerate(rate):
        spk_times = gen.poisson_generator(r, t_start=resolution, t_stop=duration, array=True)
        times.append(list(spk_times))
        ids.append(list(n * np.ones_like(times[-1])))
    ids = list(tools.utils.operations.iterate_obj_list(ids))
    tmp = [(ids[idx], round(n, rounding_precision)) for idx, n in enumerate(list(
        tools.utils.operations.iterate_obj_list(times)))]

    sl = SpikeList(tmp, list(np.unique(ids)), t_start=resolution, t_stop=duration)
    sl.round_times(resolution)

    if store:
        sl.save(store)

    return sl