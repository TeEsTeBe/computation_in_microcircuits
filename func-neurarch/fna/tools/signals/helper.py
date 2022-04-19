import numpy as np
import importlib

nest = None if importlib.util.find_spec("nest") is None else importlib.import_module("nest")

from fna.tools import check_dependency, utils
from fna.tools.utils.logger import get_logger
from fna.tools.signals.analog import AnalogSignalList, AnalogSignal
from fna.tools.signals.spikes import SpikeList

has_pyspike = check_dependency('pyspike')
if has_pyspike:
    import pyspike

logger = get_logger(__name__)


def convert_activity(initializer):
    """
    Extract recorded activity from devices, convert it to SpikeList or AnalogList
    objects and store them appropriately
    :param initializer: can be a string, or list of strings containing the relevant filenames where the
    raw data was recorded or be a gID for the recording device, if the data is still in memory
    """
    # TODO: save option!
    # if object is a string, it must be a file name; if it is a list of strings, it must be a list of filenames
    if isinstance(initializer, str) or isinstance(initializer, list):
        data = utils.data_handling.extract_data_fromfile(initializer)
        if data is not None:
            if len(data.shape) != 2:
                data = np.reshape(data, (int(len(data) / 2), 2))
            if data.shape[1] == 2:
                spk_times = data[:, 1]
                neuron_ids = data[:, 0]
                tmp = [(neuron_ids[n], spk_times[n]) for n in range(len(spk_times))]
                return SpikeList(tmp, np.unique(neuron_ids).tolist())
            else:
                neuron_ids = data[:, 0]
                times = data[:, 1]
                for nn in range(data.shape[1]):
                    if nn > 1:
                        signals = data[:, nn]
                        tmp = [(neuron_ids[n], signals[n]) for n in range(len(neuron_ids))]
                        return AnalogSignalList(tmp, np.unique(neuron_ids).tolist(), times=times)

    elif isinstance(initializer, tuple) or isinstance(initializer, int):
        status = nest.GetStatus(initializer)[0]['events']
        if len(status) == 2:
            spk_times = status['times']
            neuron_ids = status['senders']
            tmp = [(neuron_ids[n], spk_times[n]) for n in range(len(spk_times))]
            return SpikeList(tmp, np.unique(neuron_ids).tolist())
        elif len(status) > 2:
            times = status['times']
            neuron_ids = status['senders']
            idxs = np.argsort(times)
            times = times[idxs]
            neuron_ids = neuron_ids[idxs]
            rem_keys = ['times', 'senders']
            new_dict = {k: v[idxs] for k, v in status.items() if k not in rem_keys}
            analog_activity = []
            for k, v in new_dict.items():
                tmp = [(neuron_ids[n], v[n]) for n in range(len(neuron_ids))]
                analog_activity.append(AnalogSignalList(tmp, np.unique(neuron_ids).tolist(), times=times))
            return analog_activity
    else:
        print("Incorrect initializer...")


def to_pyspike(spike_list):
    """
    Convert the data in the spike_list to the format used by PySpike
    :param spike_list: SpikeList object
    :return: PySpike SpikeTrain object
    """
    assert has_pyspike, "PySpike not found.."
    bounds = spike_list.time_parameters()
    spike_trains = []
    for n_train in spike_list.id_list:
        sk_train = spike_list.spiketrains[n_train]
        pyspk_sktrain = pyspike.SpikeTrain(spike_times=sk_train.spike_times, edges=bounds)
        spike_trains.append(pyspk_sktrain)
    return spike_trains


def shotnoise_fromspikes(spike_train, q, tau, dt=0.1, t_start=None, t_stop=None, array=False, eps=1.0e-8):
    """
    Convolves the provided spike train with shot decaying exponentials yielding so called shot noise
    if the spike train is Poisson-like. Returns an AnalogSignal if array=False, otherwise (shotnoise,t)
    as numpy arrays.
    :param spike_train: a SpikeTrain object
    :param q: the shot jump for each spike
    :param tau: the shot decay time constant in milliseconds
    :param dt: the resolution of the resulting shotnoise in milliseconds
    :param t_start: start time of the resulting AnalogSignal. If unspecified, t_start of spike_train is used
    :param t_stop: stop time of the resulting AnalogSignal. If unspecified, t_stop of spike_train is used
    :param array: if True, returns (shotnoise,t) as numpy arrays, otherwise an AnalogSignal.
    :param eps: - a numerical parameter indicating at what value of the shot kernel the tail is cut.  The
    default is usually fine.
    """

    def spike_index_search(t_steps, spike_times):
        """
        For each spike, assign an index on the window timeline (t_steps)
        :param t_steps: numpy array with time points representing the binning of the time window by dt
        :param spike_times: numpy array with spike times of a spike train
        :return:
        """
        result_ = []
        spike_times.sort()
        cnt = 0
        for idx_, val in enumerate(t_steps):
            if cnt >= len(spike_times):
                break
            # check for approximate equality due to floating point fluctuations
            if np.isclose(val, spike_times[cnt], atol=0.099999):
                result_.append(idx_)
                cnt += 1
        return result_

    st = spike_train
    if t_start is not None and t_stop is not None:
        assert t_stop > t_start, "t_stop must be larger than t_start"

    # time of vanishing significance
    vs_t = -tau * np.log(eps / q)

    if t_stop is None:
        t_stop = st.t_stop

    # need to be clever with start time because we want to take spikes into account which occurred in
    # spikes_times before t_start
    if t_start is None:
        t_start = st.t_start
        window_start = st.t_start
    else:
        window_start = t_start
        if t_start > st.t_start:
            t_start = st.t_start

    t_size = int(np.round((t_stop - t_start) / dt))
    t = np.linspace(t_start, t_stop, num=t_size, endpoint=False)
    kern = q * np.exp(-np.arange(0.0, vs_t, dt) / tau)

    spike_t_idx = spike_index_search(t, st.spike_times)

    idx = np.clip(spike_t_idx, 0, len(t) - 1)
    a = np.zeros(np.shape(t), float)
    if len(spike_t_idx) > 0:
        a[idx] = 1.0
    y = np.convolve(a, kern)[0:len(t)]

    if array:
        signal_t_size = int(np.round((t_stop - window_start) / dt))
        signal_t = np.linspace(window_start, t_stop, num=signal_t_size,
                               endpoint=False)  # np.arange(window_start, t_stop, dt)
        signal_y = y[-len(signal_t):]
        return signal_y, signal_t
    else:
        result = AnalogSignal(y, dt, t_start=0.0, t_stop=t_stop - t_start)
        result.time_offset(t_start)
        if window_start > t_start:
            result = result.time_slice(window_start, t_stop)
        return result


def convert_array(array, id_list, dt=None, start=None, stop=None):
    """
    Convert a numpy array into an AnalogSignalList object
    :param array: NxT numpy array
    :param id_list:
    :param start:
    :param stop:
    :return:
    """
    assert (isinstance(array, np.ndarray)), "Provide a numpy array as input"

    if start is not None and stop is not None and dt is not None:
        # time_axis = np.arange(start, stop, dt)
        tmp = []
        for idd in range(array.shape[1]):
            for m_id, n_id in enumerate(id_list):
                tmp.append((n_id, array[m_id, idd]))
        new_AnalogSignalList = AnalogSignalList(tmp, id_list, dt=dt, t_start=start, t_stop=stop)
    else:
        new_AnalogSignalList = AnalogSignalList([], [], dt=dt, t_start=start, t_stop=stop,
                                                dims=len(id_list))

        for n, id in enumerate(np.sort(id_list)):
            try:
                id_signal = AnalogSignal(array[n, :], dt)
                new_AnalogSignalList.append(id, id_signal)
            except Exception:
                print("id %d is not in the source AnalogSignalList" % id)
    return new_AnalogSignalList


# def gather_analog_activity(parameter_set, net, t_start=None, t_stop=None):
#     """
#     Retrieve all analog activity data recorded in [t_start, t_stop]
#     :param parameter_set: global ParameterSet
#     :param net: Network object
#     :param t_start: start time
#     :param t_stop: stop time
#     :return results: organized dictionary with all analogs (can be very large!)
#     """
#     results = {}
#     for pop_n, pop in enumerate(net.populations):
#         results.update({pop.name: {}})
#         if pop.name[-5:] == 'clone':
#             pop_name = pop.name[:-6]
#         else:
#             pop_name = pop.name
#         pop_idx = parameter_set.net_pars.pop_names.index(pop_name)
#         if parameter_set.net_pars.analog_device_pars[pop_idx] is None:
#             break
#         variable_names = pop.analog_activity_names
#
#         if not pop.analog_activity:
#             results[pop.name]['recorded_neurons'] = []
#             break
#         elif isinstance(pop.analog_activity, list):
#             for idx, nn in enumerate(pop.analog_activity_names):
#                 locals()[nn] = pop.analog_activity[idx]
#                 assert isinstance(locals()[nn], AnalogSignalList), "Analog Activity should be AnalogSignalList"
#         else:
#             locals()[pop.analog_activity_names[0]] = pop.analog_activity
#
#         reversals = []
#         single_idx = np.random.permutation(locals()[pop.analog_activity_names[0]].id_list())[0]
#         results[pop.name]['recorded_neurons'] = locals()[pop.analog_activity_names[0]].id_list()
#
#         for idx, nn in enumerate(pop.analog_activity_names):
#             if (t_start is not None) and (t_stop is not None):
#                 locals()[nn] = locals()[nn].time_slice(t_start, t_stop)
#
#             time_axis = locals()[nn].time_axis()
#
#             if 'E_{0}'.format(nn[-2:]) in parameter_set.net_pars.neuron_pars[pop_idx]:
#                 reversals.append(parameter_set.net_pars.neuron_pars[pop_idx]['E_{0}'.format(nn[-2:])])
#
#             results[pop.name]['{0}'.format(nn)] = locals()[nn].as_array()
#
#     return results


def pad_array(input_array, add=10):
    """
    Pads an array with zeros along the time dimension

    :param input_array:
    :param add:
    :return:
    """
    new_shape = (input_array.shape[0], input_array.shape[1] + add)
    new_size = (new_shape[0]) * (new_shape[1])
    zero_array = np.zeros(new_size).reshape(new_shape)
    zero_array[:input_array.shape[0], :input_array.shape[1]] = input_array
    return zero_array


def make_simple_kernel(shape, width=3, height=1., resolution=1., normalize=False, **kwargs):
    """
    Simplest way to create a smoothing kernel for 1D convolution
    :param shape: {'box', 'exp', 'alpha', 'double_exp', 'gauss'}
    :param width: kernel width
    :param height: peak amplitude of the kernel
    :param resolution: time step
    :param normalize: [bool]
    :return: kernel k
    """
    # TODO load external kernel...
    x = np.arange(0., (width / resolution) + resolution, 1.)  # resolution)

    if shape == 'box':
        k = np.ones_like(x) * height

    elif shape == 'exp':
        assert 'tau' in kwargs, "for exponential kernel, please specify tau"
        tau = kwargs['tau']
        k = np.exp(-x / tau) * height

    elif shape == 'double_exp':
        assert ('tau_1' in kwargs), "for double exponential kernel, please specify tau_1"
        assert ('tau_2' in kwargs), "for double exponential kernel, please specify tau_2"

        tau_1 = kwargs['tau_1']
        tau_2 = kwargs['tau_2']
        tmp_k = (-np.exp(-x / tau_1) + np.exp(-x / tau_2))
        k = tmp_k * (height / np.max(tmp_k))

    elif shape == 'alpha':
        assert ('tau' in kwargs), "for alpha kernel, please specify tau"

        tau = kwargs['tau']
        tmp_k = ((x / tau) * np.exp(-x / tau))
        k = tmp_k * (height / np.max(tmp_k))

    elif shape == 'gauss':
        assert ('mu' in kwargs), "for Gaussian kernel, please specify mu"
        assert ('sigma' in kwargs), "for Gaussian kernel, please specify sigma"

        sigma = kwargs['sigma']
        mu = kwargs['mu']
        tmp_k = (1. / (sigma * np.sqrt(2. * np.pi))) * np.exp(- ((x - mu) ** 2. / (2. * (sigma ** 2.))))
        k = tmp_k * (height / np.max(tmp_k))

    elif shape == 'tri':
        halfwidth = width / 2
        trileft = np.arange(1, halfwidth + 2)
        triright = np.arange(halfwidth, 0, -1)  # odd number of bins
        k = np.append(trileft, triright)
        k += height

    elif shape == 'sin':
        k = np.sin(2 * np.pi * x / width * kwargs['frequency'] + kwargs['phase_shift']) * height
        k += kwargs['mean_amplitude']
    else:
        logger.warning("Kernel not implemented, please choose {'box', 'exp', 'alpha', 'double_exp', 'gauss', 'tri', "
                       "'syn'}")
        k = 0
    if normalize:
        k /= k.sum()

    return k
