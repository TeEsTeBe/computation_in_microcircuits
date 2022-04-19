import os
from time import time
import pickle
import argparse

import nest
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8, 8)

from utils import visualisation
from utils import connection_utils
from network_models.microcircuit import Microcircuit
from network_models.amorphous import AmorphousCircuit
from network_models.degreecontrolled import DegreeControlledCircuit
from network_models.smallworld import SmallWorldCircuit

nest.set_verbosity('M_ERROR')


def main(n_trials, previous_data_path):
    start_timestamp = time()
    use_previous_data = previous_data_path is not None

    name = f'degrees_per_network_{n_trials}trials'
    result_folder = os.path.join('data', 'figures', name)
    os.makedirs(result_folder, exist_ok=True)
    result_dict_path = os.path.join(result_folder, f'{name}.pkl')

    if use_previous_data:
        with open(previous_data_path, 'rb') as networks_file:
            networks = pickle.load(networks_file)
    else:
        networks = {
            'degree-controlled': {'constructor': DegreeControlledCircuit, 'degrees_per_pop': {}, 'color': 'grey'},
            'microcircuit': {'constructor': Microcircuit, 'degrees_per_pop': {}, 'color': 'green'},
            'amorphous': {'constructor': AmorphousCircuit, 'degrees_per_pop': {}, 'color': '#cf232b'},
            'small-world': {'constructor': SmallWorldCircuit, 'degrees_per_pop': {}, 'color': 'blue'},
            'degree-controlled\nw/o io': {'constructor': DegreeControlledCircuit, 'params': {'remove_io_specificity': True}, 'degrees_per_pop': {}, 'color': 'black'},
        }

    axes = None
    for net_name, net_dict in networks.items():
        print(f'\n\n{net_name}------------')
        if not use_previous_data:
            for trial in range(n_trials):
                print(f'\n{net_name}: Trial {trial+1} of {n_trials} started')
                done = False
                if 'params' in net_dict.keys():
                    network = net_dict['constructor'](disable_filter_neurons=True, **net_dict['params'])
                else:
                    network = net_dict['constructor'](disable_filter_neurons=True)

                degrees_per_pop = connection_utils.get_total_degrees_per_pop(network)
                for pop_name, degrees_array in degrees_per_pop.items():
                    if pop_name not in networks[net_name]['degrees_per_pop'].keys():
                        networks[net_name]['degrees_per_pop'][pop_name] = degrees_array
                    else:
                        networks[net_name]['degrees_per_pop'][pop_name].extend(degrees_array)

            net_degree_dict_path = os.path.join(result_folder, f'{net_name}_degrees_{n_trials}trials.pkl'.replace('/', '-'))
            with open(net_degree_dict_path, 'wb') as net_dict_file:
                pickle.dump(networks[net_name]['degrees_per_pop'], net_dict_file)

        axes = visualisation.plot_degree_distributions_per_pop(networks[net_name]['degrees_per_pop'], axes=axes,
                                                               label=net_name, color=networks[net_name]['color'], bins=50,
                                                               binrange=(0, 300))

    if not use_previous_data:
        with open(result_dict_path, 'wb') as result_file:
            pickle.dump(networks, result_file)

    duration = time() - start_timestamp
    print(f'Calculations took {duration} seconds ({duration/60.} minutes)')

    plt.ylim((0., 0.06))
    legend_order = ['microcircuit', 'amorphous', 'degree-controlled', 'degree-controlled\nw/o io', 'small-world']
    handles, labels = axes[-1][-1].get_legend_handles_labels()
    axes[-1][-1].legend([handles[labels.index(net)] for net in legend_order], [ 'data-based' if net == 'microcircuit' else net for net in legend_order])
    final_figure_path = os.path.join(result_folder, f'{name}.pdf')
    plt.savefig(final_figure_path)


def parse_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_trials', help='Number networks which will be created for every network type. Default: 100',
                        type=int, default=100)
    parser.add_argument('--previous_data_path', help='Path to data from a previous call of this script', default=None)

    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(parse_cmd()))