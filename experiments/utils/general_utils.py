import os
import itertools
import pickle
from pathlib import Path

import numpy as np
import nest

from fna.tools.signals import SpikeList
from fna.tools.signals.spikes import RandomPairs

from experiments.utils import compatability


def spikelist_from_recorder(spikedetector, stop=None, start=None):
    """ Creates a fna SpikeList from a given spike detector """

    detector_status = nest.GetStatus(spikedetector)[0]['events']
    senders = detector_status['senders']
    times = detector_status['times']
    if stop is not None:
        mask = times <= stop
        times = times[mask]
        senders = senders[mask]
    if start is not None:
        mask = times >= start
        times = times[mask]
        senders = senders[mask]
    spikes = [(neuron_id, spike_time) for neuron_id, spike_time in zip(senders, times)]
    spikelist = SpikeList(spikes, np.unique(senders), t_start=start, t_stop=stop)

    return spikelist


def align_tstart_and_tstop(spikelist1, spikelist2):
    tstart = min(spikelist1.t_start, spikelist2.t_start)
    tstop = max(spikelist1.t_stop, spikelist2.t_stop)
    for spiketrain in spikelist1:
        spiketrain.t_start = tstart
        spiketrain.t_stop = tstop
    for spiketrain in spikelist2:
        spiketrain.t_start = tstart
        spiketrain.t_stop = tstop


def get_CC_for_combinations(pop_spikelist_dict, timebin):
    """ Calculates the pairwise pearson correlatoin coefficient"""

    CC_combinations = {}
    for popname1, popname2 in itertools.combinations(pop_spikelist_dict.keys(), r=2):
        spikelist1 = pop_spikelist_dict[popname1]
        spikelist2 = pop_spikelist_dict[popname2]
        align_tstart_and_tstop(spikelist1, spikelist2)
        pairs_generator = RandomPairs(spikelist1, spikelist2, no_silent=True, no_auto=True)
        CC_combinations[f'{popname1}-{popname2}'] = spikelist1.pairwise_pearson_corrcoeff(1000, pairs_generator, time_bin=timebin, all_coef=True)

    return CC_combinations


def get_spike_statistics(population_spikedetector_dict):
    """ Calculates different spiking statistics """

    print('\ncalculating spike statistics ...')
    spike_lists = {}
    for populationname, detector in population_spikedetector_dict.items():
        spike_lists[populationname] = spikelist_from_recorder(detector)

    spike_statistics = {}
    for populationname, spikelist in spike_lists.items():
        spike_statistics[populationname] = {
            'rates': spikelist.mean_rates(),
            'cv': spikelist.cv_isi(),
            'cv_kl': spikelist.cv_kl(),
            'local_variation': spikelist.local_variation(),
            'lv_r': spikelist.local_variation_revised(),
            'fano_factor_isi': spikelist.fano_factors_isi(),
            'isi_entropy': spikelist.isi_entropy(),
            'isi_5p': spikelist.isi_5p(),
        }
        spike_statistics[populationname]['cc'] = {}
        spike_statistics[populationname]['fano_factor'] = {}
        for timebin in np.arange(1., 21.):
            spike_statistics[populationname]['cc'][f'timebin{timebin}'] = spikelist.pairwise_pearson_corrcoeff(1000, time_bin=timebin, all_coef=True)
            spike_statistics[populationname]['fano_factor'][f'timebin{timebin}'] = spikelist.fano_factors(time_bin=timebin)

    spike_statistics['cc_combinations'] = {}
    for timebin in np.arange(1., 21.):
        spike_statistics['cc_combinations'][f'timebin{timebin}'] = get_CC_for_combinations(spike_lists, timebin=timebin)

    print('\tdone!')

    return spike_statistics


def order_array_by_ids(array_to_order, n_possible_ids, ids):
    """
    Orders an array (for example spike trains of neurons) by the given ids (of the neurons).
    Needs the number of possible (neuron) ids, because some ids could be missing (neurons may not have
    fired), but they should be in the resulting list as well.

    Parameters
    ----------
    array_to_order: ndarray of floats
        ndarray with spike times
    n_possible_ids: int
        number of possible ids
    ids: ndarray of int
        ids of the objects to which the elements in the array_to_order belong

    Returns
    -------
    ndarray
        spike trains (ndarrays) for each neuron

    Examples
    --------
    >>> spike_times = np.array([10.2, 20.1, 30.1])
    >>> ids = np.array([2, 1, 1])
    >>> order_array_by_ids(spike_times, 3, ids)
    [array([20.1, 30.1]), array([10.2]), array([], dtype=float64)]
    """

    if len(array_to_order) == 0:
        print('Array to order is empty!')
        return None
    else:
        spk_times_list = [np.array([]) for _ in range(n_possible_ids)]
        neurons = np.unique(ids)
        new_ids = ids - min(ids)

        for i, n in enumerate(neurons):
            idx = np.where(ids == n)[0]
            spk_times_list[new_ids[idx[0]]] = array_to_order[idx]

        spk_times_array = np.array(spk_times_list)
        # spk_times_array = np.hstack((spk_times_array, spk_times_array[:, -1]))

        return spk_times_array


def create_spike_detectors(populations, start=0., stop=None):
    spike_detectors = {}

    params = {'start': start}
    if stop is not None:
        params['stop'] = stop

    for pop_name in populations.keys():
        spike_detectors[pop_name] = nest.Create(compatability.spike_detector_naming, params=params)

    for pop_name, pop in populations.items():
        nest.Connect(pop, spike_detectors[pop_name])

    return spike_detectors


def get_data_dir():
    return os.path.join(Path(__file__).parent.parent.resolve(), 'data')


def get_runname_and_results_folder(argument_dict, runtitle, parent_name, group_name, ok_if_folder_exists=False, maxlen=250):
    runname = create_name_from_arguments(argument_dict, runtitle)[:maxlen]
    results_folder = os.path.join(get_data_dir(), parent_name, group_name, runname)
    os.makedirs(results_folder, exist_ok=ok_if_folder_exists)
    with open(os.path.join(results_folder, 'parameters.pkl'), 'wb') as parameters_file:
        pickle.dump(argument_dict, parameters_file)

    print(f'# {runtitle}')
    print('## Parameters:')
    for key, value in argument_dict.items():
        print(f'- {key}:\t{value}')

    return runname, results_folder


def create_name_from_arguments(argument_dict, runtitle):
    argument_strings = []
    shortnames = {
        'network_name': 'net',
        'reset_neurons': 'res_neur',
        'reset_synapses': 'res_syn',
        'steps_per_trial': 'steps',
        'discard_steps': 'discard',
        'train_trials': 'train',
        'test_trials': 'test',
        'step_duration': 'steps',
        'raster_plot_duration': 'raster_dur',
        'num_threads': 'threads',
        'activate_M_ion_channels': 'MIon',
    }
    keys_to_ignore = ['runtitle', 'ok_if_folder_exists']

    for arg, val in argument_dict.items():
        if arg in shortnames.keys():
            arg = shortnames[arg]
        if arg not in keys_to_ignore:
            argument_strings.append(f'{arg}={val}')
    runname = f'{runtitle}_' + '_'.join(argument_strings)

    return runname


def get_parameters_from_nest(entity_ids, parameter_keys):
    parameter_values = nest.GetStatus(entity_ids, keys=parameter_keys)
    saved_parameters = []
    for values_per_entity in parameter_values:
        saved_parameters.append(dict([(key, value) for key, value in zip(parameter_keys, values_per_entity)]))

    return saved_parameters


def get_neuron_states(populations, neuron_pars):
    neuron_states = {}
    for pop_name, neuron_ids in populations.items():
        parameter_keys = neuron_pars[pop_name].keys()
        neuron_states[pop_name] = get_parameters_from_nest(neuron_ids, parameter_keys)

    return neuron_states


def get_synapse_states(connections_ids_from_to, syn_params_from_to):
    synapse_states = {}
    for src_pop_name, trg_pop_to_ids in connections_ids_from_to.items():
        synapse_states[src_pop_name] = {}
        for trg_pop_name, connection_ids in trg_pop_to_ids.items():
            parameter_keys = list(syn_params_from_to[src_pop_name[-3:]][trg_pop_name[-3:]].keys())
            parameter_keys.remove(compatability.syn_model_naming)
            parameter_keys.append('weight')
            parameter_keys.append('x')
            synapse_states[src_pop_name][trg_pop_name] = get_parameters_from_nest(connection_ids, parameter_keys)

    return synapse_states


def get_dict_combinations(dictionary):
    """
    Transforms a dictionary with lists as dict values into a list of all combinations of the list items.

    Parameters
    ----------
    dictionary: dict
        dictionary with list as values

    Returns
    -------
    list
        list of dictionaries with all combinations

    Examples
    --------
    >>> mydict = {'a': [0, 1], 'b': [2, 3], 'c': ['xyz']}
    >>> get_dict_combinations(mydict)
    [{'a': 0, 'b': 2, 'c': 'xyz'}, {'a': 0, 'b': 3, 'c': 'xyz'}, {'a': 1, 'b': 2, 'c': 'xyz'}, {'a': 1, 'b': 3, 'c': 'xyz'}]


    """

    parameter_keys, parameter_values = zip(*dictionary.items())
    dict_combinations = [dict(zip(parameter_keys, v)) for v in itertools.product(*parameter_values)]

    return dict_combinations


def simulate_with_reset(network, simtime, trial_simtime, reset_neurons, reset_synapses):
    initial_neuron_states = get_neuron_states(network.populations, network.neuron_pars)
    initial_synapse_states = get_synapse_states(network.conn_ids_from_to, network.syn_params_from_to)
    remaining_simtime = simtime
    time_already_simulated = 0.
    while remaining_simtime > 0.:
        nest.Simulate(min(remaining_simtime, trial_simtime))
        remaining_simtime -= trial_simtime
        time_already_simulated += trial_simtime
        if reset_neurons:
            for pop_name, neuron_ids in network.populations.items():
                nest.SetStatus(neuron_ids, initial_neuron_states[pop_name])
        if reset_synapses:
            for src_pop_name, trg_pop_to_ids in network.conn_ids_from_to.items():
                for trg_pop_name, connection_ids in trg_pop_to_ids.items():
                    nest.SetStatus(connection_ids, initial_synapse_states[src_pop_name][trg_pop_name])
        print(
            f'{time_already_simulated} ms of {simtime} ms simulated ({round(100 * time_already_simulated / simtime, 2)}%)',
            flush=True)


def get_multimeter_of_neuron_sample(network, tasks):
    random_neuron_sample = []
    n_neuron_sample = 3
    for pop_name, neuron_ids in network.populations.items():
        random_neuron_sample.extend(np.random.choice(neuron_ids, size=n_neuron_sample, replace=False))
    random_neuron_sample = sorted(random_neuron_sample)
    sample_multimeter = nest.Create('multimeter',
                                    {'record_from': ['V_m'], 'stop': tasks.steps_per_trial * tasks.step_duration})
    nest.Connect(sample_multimeter, random_neuron_sample)
    return sample_multimeter


def get_pickled_data_from_subfolders(parentfolder, contains_string=None, filename='results.pkl', none_if_doesnt_exist=False):
    runnames = os.listdir(parentfolder)
    file_paths = [os.path.join(parentfolder, name, filename) for name in runnames]
    if contains_string is not None:
        file_paths = [p for p in file_paths if contains_string in p]

    unpickled_data = []
    for path in file_paths:
        if os.path.exists(path) or not none_if_doesnt_exist:
            with open(path, 'rb') as pickled_file:
                unpickled_data.append(pickle.load(pickled_file))
        else:
            print(f"{path} doesn't exist!")
            unpickled_data.append(None)

    return unpickled_data


def get_aggregated_result(result_dicts, taskname, readout_name, metric_name, accumulation_function=np.nanmean):
    values = []
    for result in result_dicts:
        values.append(result[taskname][readout_name][metric_name])

    return accumulation_function(values)


def get_full_aggregated_dict(list_of_dicts, accumulation_function=np.nanmean):
    accumulated_dict = {}
    dict_keys = list(list_of_dicts[0].keys())
    for key in dict_keys:
        dict_values = [dict_[key] for dict_ in list_of_dicts]
        if isinstance(dict_values[0], dict):
            accumulated_value = get_full_aggregated_dict(dict_values, accumulation_function=accumulation_function)
        elif hasattr(dict_values[0], '__len__') and not isinstance(dict_values[0], str):
            dict_values = [accumulation_function(x) for x in dict_values]
            accumulated_value = accumulation_function(dict_values)
        else:
            accumulated_value = accumulation_function(dict_values)

        accumulated_dict[key] = accumulated_value

    return accumulated_dict


def flatten_dict(dict_to_flatten, prefix=''):
    flattened_dict = {}
    for key, value in dict_to_flatten.items():
        if isinstance(value, dict):
            flattened_value_dict = flatten_dict(value, prefix=f'{key}_')
            for sub_key, sub_value in flattened_value_dict.items():
                flattened_dict[f'{prefix}{sub_key}'] = sub_value

        else:
            flattened_dict[f'{prefix}{key}'] = value

    return flattened_dict


def standard_error_of_mean(array):
    return np.nanstd(array) / np.sqrt(len(array))
