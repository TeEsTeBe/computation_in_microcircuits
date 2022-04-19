import os
import sys

import numpy as np
import pandas as pd

from utils.general_utils import get_pickled_data_from_subfolders, get_aggregated_result, get_data_dir


if __name__ == "__main__":
    if len(sys.argv) > 1:
        group_prefix = f'{sys.argv[1]}_'
    else:
        group_prefix = ''

    # data_base_path = '/home/schultetobrinke/projects/hm2006/repos/hm2006/experiments/data/hambach'
    data_dir = get_data_dir()
    net_to_groupname_spikes = {
        'microcircuit': f'{group_prefix}microcircuit_spikes',
        'amorphous': f'{group_prefix}amorphous_spikes',
        'smallworld': f'{group_prefix}smallworld_spikes',
        'degreecontrolled': f'{group_prefix}degreecontrolled_spikes',
        'degreecontrolled_no_io_specificity': f'{group_prefix}degreecontrolled_no_io_specificity_spikes',
        'microcircuit_random_dynamics': f'{group_prefix}microcircuit_random_dynamics_spikes',
        'microcircuit_static': f'{group_prefix}microcircuit_static_spikes',
    }

    net_to_groupname_rates = {
        'microcircuit': f'{group_prefix}microcircuit_rates',
        'amorphous': f'{group_prefix}amorphous_rates',
        'smallworld': f'{group_prefix}smallworld_rates',
        'degreecontrolled': f'{group_prefix}degreecontrolled_rates',
        'degreecontrolled_no_io_specificity': f'{group_prefix}degreecontrolled_no_io_specificity_rates',
        'microcircuit_random_dynamics': f'{group_prefix}microcircuit_random_dynamics_rates',
        'microcircuit_static': f'{group_prefix}microcircuit_static_rates',
    }

    avg_task_results = {}
    avg_task_results_rounded = {}

    for name, groupname_spikes in net_to_groupname_spikes.items():

        # results_path_spike_pattern = os.path.join(data_base_path, foldername_spk_pattern, f'{name}_only_results')
        results_path_spike_pattern = os.path.join(data_dir, 'task_results', groupname_spikes)
        result_dicts_spike_pattern = get_pickled_data_from_subfolders(results_path_spike_pattern)

        groupname_rates = net_to_groupname_rates[name]
        # results_path_rates = os.path.join(data_base_path, foldername_rates, f'{name}_only_results')
        results_path_rates = os.path.join(data_dir, 'task_results', groupname_rates)
        result_dicts_rates = get_pickled_data_from_subfolders(results_path_rates)

        avg_task_results[name] = {
            # other
            'classification_S1_l23' : get_aggregated_result(result_dicts_spike_pattern, 'spike_pattern_classification_S1', readout_name='l23_exc_nnls', metric_name='kappa_test'),
            'classification_S2_l23' : get_aggregated_result(result_dicts_spike_pattern, 'spike_pattern_classification_S2', readout_name='l23_exc_nnls', metric_name='kappa_test'),
            'classification_S1_l5' : get_aggregated_result(result_dicts_spike_pattern, 'spike_pattern_classification_S1', readout_name='l5_exc_nnls', metric_name='kappa_test'),
            'classification_S2_l5' : get_aggregated_result(result_dicts_spike_pattern, 'spike_pattern_classification_S2', readout_name='l5_exc_nnls', metric_name='kappa_test'),

            # memory
            'delayed_classification_S1_l23' : get_aggregated_result(result_dicts_spike_pattern, 'delayed_spike_pattern_classification_S1', readout_name='l23_exc_nnls', metric_name='kappa_test'),
            'delayed_classification_S2_l23' : get_aggregated_result(result_dicts_spike_pattern, 'delayed_spike_pattern_classification_S2', readout_name='l23_exc_nnls', metric_name='kappa_test'),
            'delayed_classification_S1_l5' : get_aggregated_result(result_dicts_spike_pattern, 'delayed_spike_pattern_classification_S1', readout_name='l5_exc_nnls', metric_name='kappa_test'),
            'delayed_classification_S2_l5' : get_aggregated_result(result_dicts_spike_pattern, 'delayed_spike_pattern_classification_S2', readout_name='l5_exc_nnls', metric_name='kappa_test'),

            # nonlinear
            'xor_l23' : get_aggregated_result(result_dicts_spike_pattern, 'xor_spike_pattern', readout_name='l23_exc_nnls', metric_name='kappa_test'),
            'r1/r2_l23' : get_aggregated_result(result_dicts_rates, 'r1/r2', readout_name='l23_exc_nnls', metric_name='cc_test'),
            '(r1-r2)^2_l23' : get_aggregated_result(result_dicts_rates, '(r1-r2)^2', readout_name='l23_exc_nnls', metric_name='cc_test'),
            'xor_l5' : get_aggregated_result(result_dicts_spike_pattern, 'xor_spike_pattern', readout_name='l5_exc_nnls', metric_name='kappa_test'),
            'r1/r2_l5' : get_aggregated_result(result_dicts_rates, 'r1/r2', readout_name='l5_exc_nnls', metric_name='cc_test'),
            '(r1-r2)^2_l5' : get_aggregated_result(result_dicts_rates, '(r1-r2)^2', readout_name='l5_exc_nnls', metric_name='cc_test'),
        }
        avg_task_results_rounded[name] = {}
        for task, result in avg_task_results[name].items():
            avg_task_results_rounded[name][task] = round(result, 3)

    task_types = {
        'other' : ['classification_S1_l23', 'classification_S2_l23', 'classification_S2_l5', 'classification_S2_l5'],
        'memory' : ['delayed_classification_S1_l23', 'delayed_classification_S2_l23', 'delayed_classification_S2_l5', 'delayed_classification_S2_l5'],
        'nonlinear' : ['xor_l23', 'r1/r2_l23', '(r1-r2)^2_l23', 'xor_l5', 'r1/r2_l5', '(r1-r2)^2_l5'],
        'all': [],
    }
    for type, tasknames in task_types.items():
        if type != 'all':
            task_types['all'].extend(tasknames)

    task_differences = {}
    task_differences_percent_rounded = {}
    for networkname, result_dict in avg_task_results.items():
        print(f'\n\n# {networkname}\n')
        task_differences[networkname] = {}
        task_differences_percent_rounded[networkname] = {}
        for taskname, result in result_dict.items():
            task_result_mc = avg_task_results['microcircuit'][taskname]
            task_result_net = avg_task_results[networkname][taskname]
            difference = (task_result_net - task_result_mc) / task_result_mc
            task_differences[networkname][taskname] = difference
            task_differences_percent_rounded[networkname][taskname] = round(difference*100, 1)
            print(f'## {taskname}')
            print(f'{task_result_net} ({difference*100}%)\tMC: {task_result_mc}')
            print(f'{round(task_result_net, 3)} ({round(difference*100, 1)}%)\tMC: {round(task_result_mc, 3)}\n')

        for tasktype, tasknames in task_types.items():
            result_list = []
            result_list_mc = []
            for task in tasknames:
                result_list.append(avg_task_results[networkname][task])
                result_list_mc.append(avg_task_results['microcircuit'][task])
            avg_result = np.mean(result_list)
            avg_result_mc = np.mean(result_list_mc)
            difference = (avg_result - avg_result_mc) / avg_result_mc
            task_differences[networkname][tasktype] = difference
            task_differences_percent_rounded[networkname][tasktype] = round(difference * 100, 1)
            avg_task_results[networkname][tasktype] = avg_result
            avg_task_results_rounded[networkname][tasktype] = round(avg_result, 3)

            print(f'## avg {tasktype} tasks')
            print(f'{avg_result} ({difference*100}%)\tMC: {avg_result_mc}')
            print(f'{round(avg_result, 3)} ({round(difference*100,1)}%)\tMC: {round(avg_result_mc, 3)}')

    pd.DataFrame(task_differences).to_csv(os.path.join(data_dir, f'{group_prefix}task_differences.csv') ,sep=',')
    pd.DataFrame(task_differences_percent_rounded).to_csv(os.path.join(data_dir, f'{group_prefix}task_differences_percent_rounded.csv'), sep=',')
    pd.DataFrame(avg_task_results).to_csv(os.path.join(data_dir, f'{group_prefix}average_task_results.csv'), sep=',')
    pd.DataFrame(avg_task_results_rounded).to_csv(os.path.join(data_dir, f'{group_prefix}average_task_results_rounded.csv'), sep=',')
