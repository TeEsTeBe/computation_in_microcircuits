import os
import time
import pickle
import argparse

import nest
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from experiments.tasks.classification_task_evaluator import ClassificationTaskEvaluator
from experiments.tasks.rate_task_evaluator import RateTaskEvaluator
from experiments.utils import general_utils, visualisation, connection_utils
from experiments.network_models.microcircuit import Microcircuit
from experiments.network_models.amorphous import AmorphousCircuit
from experiments.network_models.smallworld import SmallWorldCircuit
from experiments.network_models.degreecontrolled import DegreeControlledCircuit
from experiments.tasks.spike_pattern_classification import SpikePatternClassification
from experiments.tasks.firing_rate_tasks import FiringRateTask


class SpikePatternTaskRunner:

    def __init__(self, network_name='microcircuit', N=560, S_rw=119.3304, S1=0.014, S2=0.033, reset_neurons=False, reset_synapses=False,
                 steps_per_trial=15, discard_steps=0, train_trials=1500, test_trials=300, step_duration=30.,
                 runtitle='defaultname', raster_plot_duration=450., max_delay=1, spike_statistics_duration=10000., num_threads=8,
                 gM_exc=100., gM_inh=0., ok_if_folder_exists=False, rate_tasks=False, input_dimensions=40, freeze_last_input=False,
                 start_s2=0., group_name='defaultgroupname', neuron_model=None, disable_conductance_noise=False,
                 vt_l23exc=None, vt_l23inh=None, vt_l4exc=None, vt_l4inh=None, vt_l5exc=None, vt_l5inh=None):

        argument_dict = locals().copy()
        del argument_dict['self']
        spikes_or_rates = 'rates' if rate_tasks else 'spikes'
        self.group_name = f'{group_name}_{network_name}_{spikes_or_rates}'
        self.runname, self.results_folder = general_utils.get_runname_and_results_folder(argument_dict=argument_dict,
                                                                                         runtitle=runtitle,
                                                                                         parent_name='task_results',
                                                                                         group_name=self.group_name,
                                                                                         ok_if_folder_exists=ok_if_folder_exists)
        self.network_name = network_name
        self.neuron_model = neuron_model
        self.N = N
        self.S_rw = S_rw
        self.S1 = S1
        self.S2 = S2
        self.reset_neurons = reset_neurons
        self.reset_synapses = reset_synapses
        self.steps_per_trial = steps_per_trial
        self.discard_steps = discard_steps
        if self.steps_per_trial == 1:
            self.discard_steps = max(1, discard_steps)  # need at least one discarded step because of delay task
        self.train_trials = train_trials
        self.test_trials = test_trials
        self.step_duration = step_duration
        self.runtitle = runtitle
        self.raster_plot_duration = raster_plot_duration
        self.spike_statistics_duration = spike_statistics_duration
        self.max_delay = max_delay
        self.num_threads = num_threads
        self.gM_exc = gM_exc
        self.gM_inh = gM_inh
        self.ok_if_folder_exists = ok_if_folder_exists
        self.rate_tasks = rate_tasks
        self.input_dimensions = input_dimensions
        self.freeze_last_input = freeze_last_input
        self.start_s2 = start_s2
        self.disabel_conductance_noise = disable_conductance_noise
        self.vt_l23exc = vt_l23exc
        self.vt_l23inh = vt_l23inh
        self.vt_l4exc = vt_l4exc
        self.vt_l4inh = vt_l4inh
        self.vt_l5exc = vt_l5exc
        self.vt_l5inh = vt_l5inh

    def create_network(self):
        print('Creating network ...')
        network_parameters = {
            'N': self.N,
            'S_rw': self.S_rw,
            'gM_exc': self.gM_exc,
            'gM_inh': self.gM_inh,
            'multimeter_sample_interval': self.step_duration,
            'neuron_model': self.neuron_model,
            'disable_conductance_noise': self.disabel_conductance_noise,
            'vt_l23exc': self.vt_l23exc,
            'vt_l23inh': self.vt_l23inh,
            'vt_l4exc': self.vt_l4exc,
            'vt_l4inh': self.vt_l4inh,
            'vt_l5exc': self.vt_l5exc,
            'vt_l5inh': self.vt_l5inh,
        }
        if self.network_name == 'microcircuit':
            network = Microcircuit(**network_parameters)
        elif self.network_name == 'microcircuit_static':
            network_parameters['static_synapses'] = True
            network = Microcircuit(**network_parameters)
        elif self.network_name == 'microcircuit_random_dynamics':
            network_parameters['random_synaptic_dynamics'] = True
            network = Microcircuit(**network_parameters)
        elif self.network_name == 'amorphous':
            network = AmorphousCircuit(**network_parameters)
        elif self.network_name == 'degreecontrolled':
            network = DegreeControlledCircuit(**network_parameters)
        elif self.network_name == 'degreecontrolled_no_io_specificity':
            network_parameters['remove_io_specificity'] = True
            network = DegreeControlledCircuit(**network_parameters)
        elif self.network_name == 'smallworld':
            network_parameters['random_weight_when_unconnected'] = True
            network = SmallWorldCircuit(**network_parameters)
        elif self.network_name == 'smallworld_norandomweight':
            network_parameters['random_weight_when_unconnected'] = False
            network = SmallWorldCircuit(**network_parameters)
        else:
            raise ValueError(f'{self.network_name} not implemented!')

        return network

    def init_nest(self):
        nest.ResetKernel()
        nest.SetKernelStatus({"local_num_threads": self.num_threads, 'print_time': False})

    def setup_task(self, network):
        print('Setting up tasks ...')
        if self.rate_tasks:
            tasks = FiringRateTask(steps_per_trial=self.steps_per_trial, step_duration=self.step_duration,
                                   discard_steps=self.discard_steps, train_trials=self.train_trials,
                                   test_trials=self.test_trials, dimensions=self.input_dimensions, max_delay=self.max_delay)
        else:
            tasks = SpikePatternClassification(steps_per_trial=self.steps_per_trial, step_duration=self.step_duration,
                                               discard_steps=self.discard_steps, train_trials=self.train_trials,
                                               test_trials=self.test_trials, dimensions=self.input_dimensions,
                                               freeze_last_input=self.freeze_last_input, start_s2=self.start_s2,
                                               max_delay=self.max_delay)
        network.connect_input_stream1(tasks.spike_generators_s1, scaling_factor=self.S1)
        network.connect_input_stream2(tasks.spike_generators_s2, scaling_factor=self.S2)

        self.save_task_data_to_disk(tasks)

        return tasks

    def simulate(self, tasks, network):
        start_sim = time.time()
        if self.reset_synapses or self.reset_neurons:
            trial_simtime = tasks.step_duration * tasks.steps_per_trial
            general_utils.simulate_with_reset(network=network, simtime=tasks.simtime, trial_simtime=trial_simtime,
                                              reset_neurons=self.reset_neurons, reset_synapses=self.reset_synapses)
        else:
            nest.Simulate(tasks.simtime)
        simtime_real = time.time() - start_sim

        print(f'\nSimulation time: {simtime_real} seconds ({round(simtime_real/60, 2)} minutes; {round(simtime_real/60/60, 4)} hours)\n')

        return simtime_real

    def create_plots(self, simtime, network, results_folder, sample_multimeter, spike_detectors, degrees_per_pop):
        print('Creating plots ...')
        usable_statistics_duration = min(simtime, self.spike_statistics_duration)
        visualisation.plot_neuron_sample_traces(results_folder, sample_multimeter)
        visualisation.store_rasterplot(results_folder, spike_detectors, xlim=(0, min(simtime, self.raster_plot_duration)))
        visualisation.store_rasterplot(results_folder, spike_detectors, xlim=(0, min(simtime, usable_statistics_duration)))
        if network.network_type == 'smallworld':
            for duration in [usable_statistics_duration, self.raster_plot_duration]:
                for cut_ms in [100., 0.]:
                    fig, ax = plt.subplots()
                    rates = visualisation.firing_rate_hist(ax=ax, spk_det=spike_detectors['all_exc'], sim_time=duration,
                                                           num_neurons=network.N,title='Small World Network', cut_first_ms=cut_ms)
                    plt.savefig(os.path.join(results_folder, f'firingrates_{duration}ms_cut{cut_ms}ms.pdf'))
        else:
            visualisation.store_firing_rate_histograms(network, self.raster_plot_duration, results_folder, spike_detectors, cut_first_ms=100.)
            visualisation.store_firing_rate_histograms(network, usable_statistics_duration, results_folder, spike_detectors, cut_first_ms=100.)
            visualisation.store_firing_rate_histograms(network, self.raster_plot_duration, results_folder, spike_detectors, cut_first_ms=0.)
            visualisation.store_firing_rate_histograms(network, usable_statistics_duration, results_folder, spike_detectors, cut_first_ms=0.)

        axes = visualisation.plot_degree_distributions_per_pop(degrees_per_pop, label=network.network_type)
        plt.savefig(os.path.join(results_folder, 'degree_distributions.pdf'))

    def save_task_data_to_disk(self, tasks):
        with open(os.path.join(self.results_folder, 'task_targets.pkl'), 'wb') as target_file:
            pickle.dump(tasks.targets, target_file)
        np.save(os.path.join(self.results_folder, 'input_values_s1.npy'), tasks.input_values_s1)
        np.save(os.path.join(self.results_folder, 'input_values_s2.npy'), tasks.input_values_s2)
        with open(os.path.join(self.results_folder, 'input_spikes_per_generator_s1.pkl'), 'wb') as spikes_s1_file:
            pickle.dump(tasks.spiketimes_per_generator_s1, spikes_s1_file)
        with open(os.path.join(self.results_folder, 'input_spikes_per_generator_s2.pkl'), 'wb') as spikes_s2_file:
            pickle.dump(tasks.spiketimes_per_generator_s2, spikes_s2_file)

    def get_state_matrices(self, network, tasks):
        network.calculate_state_matrices()
        full_statematrices = network.get_state_matrices()
        statematrices = network.get_state_matrices(discard_steps=tasks.discard_steps,
                                                   steps_per_trial=tasks.steps_per_trial)
        for matname, mat in full_statematrices.items():
            np.save(os.path.join(self.results_folder, f'statemat_{matname}.npy'), mat)
        for matname, mat in statematrices.items():
            np.save(os.path.join(self.results_folder, f'sampled_statemat_{matname}.npy'), mat)
        return statematrices

    def evaluate_tasks(self, statematrices, tasks):
        print('Evaluating tasks ...')
        if self.rate_tasks:
            task_evaluator = RateTaskEvaluator(statematrices, tasks.targets, tasks.train_trials, tasks.test_trials)
        else:
            task_evaluator = ClassificationTaskEvaluator(statematrices, tasks.targets, tasks.train_trials,
                                                         tasks.test_trials)
        task_eval_start = time.time()
        results_dict = task_evaluator.evaluate_all_tasks()
        task_eval_time = time.time() - task_eval_start
        print(f'Task evaluation time: {task_eval_time} seconds ({round(task_eval_time/60, 2)} minutes; {round(task_eval_time/60/60, 4)} hours)')
        with open(os.path.join(self.results_folder, 'results.pkl'), 'wb') as results_file:
            pickle.dump(results_dict, results_file)

        return task_eval_time

    def calculate_spike_statistics(self, spike_detectors):
        print('Calculating spike statistics ...')
        spike_stats_start = time.time()
        spike_statistics = general_utils.get_spike_statistics(spike_detectors)
        spike_stats_time = time.time() - spike_stats_start
        print(f'Spike statistics calculation time: {spike_stats_time} seconds ({round(spike_stats_time / 60, 2)} minutes; {round(spike_stats_time / 60 / 60, 4)} hours)')

        with open(os.path.join(self.results_folder, 'spike_statistics.pkl'), 'wb') as stats_file:
            pickle.dump(spike_statistics, stats_file)

        return spike_stats_time

    def calculate_graph_statistics(self, network):
        print('Calculating graph statistics ...')

        graph_stats_start = time.time()
        network_graph = connection_utils.get_network_graph(network)
        graph_statistics = {
            'average_clustering': nx.average_clustering(network_graph),
            'average_shortest_path_length': nx.average_shortest_path_length(network_graph)
        }
        graph_stats_time = time.time() - graph_stats_start
        print(f'Graph statistics calculation time: {graph_stats_time} seconds ({round(graph_stats_time / 60, 2)} minutes; {round(graph_stats_time / 60 / 60, 4)} hours)')

        with open(os.path.join(self.results_folder, 'network_graph_statistics.pkl'), 'wb') as graph_stats_file:
            pickle.dump(graph_statistics, graph_stats_file)

        print(f"clustering: {graph_statistics['average_clustering']}")
        print(f"shortest path length: {graph_statistics['average_shortest_path_length']}")

        return graph_stats_time

    def store_input_data_per_neuron(self, network):
        input_data_per_neuron = connection_utils.get_input_data_per_neuron(network)
        with open(os.path.join(self.results_folder, 'input_data_per_neuron.pkl'), 'wb') as input_data_file:
            pickle.dump(input_data_per_neuron, input_data_file)

    def get_and_store_degrees_per_pop(self, network):
        degrees_per_pop = connection_utils.get_total_degrees_per_pop(network)
        with open(os.path.join(self.results_folder, 'degrees_per_pop.pkl'), 'wb') as degrees_file:
            pickle.dump(degrees_per_pop, degrees_file)

        return degrees_per_pop

    def run(self):

        start_timestamp = time.time()
        self.init_nest()

        # setup network and tasks
        network = self.create_network()
        tasks = self.setup_task(network)

        # detectors for additional plots
        sample_multimeter = general_utils.get_multimeter_of_neuron_sample(network, tasks)
        spike_detectors = general_utils.create_spike_detectors(network.populations,
                                                               stop=max(self.raster_plot_duration, self.spike_statistics_duration))
        # simulation
        simtime_real = self.simulate(tasks, network)

        statematrices = self.get_state_matrices(network, tasks)

        # evaluate the tasks (training readouts and calculating the kappa coefficient)
        task_eval_time = self.evaluate_tasks(statematrices, tasks)

        degrees_per_pop = self.get_and_store_degrees_per_pop(network)
        self.store_input_data_per_neuron(network)

        self.create_plots(tasks.simtime, network, self.results_folder, sample_multimeter, spike_detectors, degrees_per_pop)

        # calculate statistics
        spike_stats_time = self.calculate_spike_statistics(spike_detectors)
        graph_stats_time = self.calculate_graph_statistics(network)

        total_runtime = time.time() - start_timestamp
        print('\n## Runtimes ')
        print(f'- simulation time: {simtime_real} seconds ({round(simtime_real/60, 2)} minutes; {round(simtime_real/60/60, 4)} hours)')
        print(f'- task evaluation time: {task_eval_time} seconds ({round(task_eval_time/60, 2)} minutes; {round(task_eval_time/60/60, 4)} hours)')
        print(f'- spike statistics calculation time: {spike_stats_time} seconds ({round(spike_stats_time/60, 2)} minutes; {round(spike_stats_time/60/60, 4)} hours)')
        print(f'- graph statistics calculation time: {graph_stats_time} seconds ({round(graph_stats_time/60, 2)} minutes; {round(graph_stats_time/60/60, 4)} hours)')
        print(f'- total run time: {total_runtime} seconds ({round(total_runtime/60, 2)} minutes; {round(total_runtime/60/60, 4)} hours)')


def parse_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument('--network_name', help='Network model, which should solve the task. Possible values: '
                                               'microcircuit, amorphous, degreecontrolled, '
                                               'degreecontrolled_no_io_specificity, smallworld, smallworld_norandomweight,'
                                               'microcircuit_static, microcircuit_random_dynamics',
                        default='microcircuit')
    parser.add_argument('--N', help='Number of neurons in the network. Default: 560', type=int, default=560)
    parser.add_argument('--S_rw', help="Scaling factor for the recurrent weights. Default: 66825/N (static synapses: 66825/N/73)", type=float, default=None)
    parser.add_argument('--S1', help="Scaling factor for the weights of the connections from the first input stream",
                        type=float, default=None)
    parser.add_argument('--S2', help="Scaling factor for the weights of the connections from the second input stream",
                        type=float, default=None)
    parser.add_argument('--reset_neurons', help='If set the neuron states are reset after every trial',
                        action='store_true', default=False)
    parser.add_argument('--reset_synapses', help='If set the synapse states are reset after every trial',
                        action='store_true', default=False)
    parser.add_argument('--steps_per_trial', help='Defines how many steps there are per trial. Default: 15', type=int, default=15)
    parser.add_argument('--discard_steps', help='Defines how many steps are discarded at the beginning.', type=int,
                        default=0)
    parser.add_argument('--train_trials', help='Number of trials used to train the readouts. Default: 1500', type=int, default=1500)
    parser.add_argument('--test_trials', help='Number of trials used to test the readouts. Default: 300', type=int, default=300)
    parser.add_argument('--step_duration', help='Duration of a single step in milliseconds. Default: 30.', type=float, default=30.)
    parser.add_argument('--raster_plot_duration', help='Defines for how long (in milliseconds) spikes are plottet. Default: 450.',
                        type=float, default=450.)
    parser.add_argument('--max_delay', help='Maximum delay (in steps) for memory tasks', type=int, default=1)
    parser.add_argument('--spike_statistics_duration', help='Defines for how long (in milliseconds) spikes are detected'
                                                            'and used for spike statistic calculations. Default: 10000.', type=float,
                        default=10000.)
    parser.add_argument('--runtitle', help='Title/name of the run. This is added to the results folder name', type=str,
                        default='default-title')
    parser.add_argument('--num_threads', help='Number of threads used by NEST.', type=int, default=8)
    parser.add_argument('--gM_exc', help='Peak conductance of Mainen potassium ion channel for excitatory neurons. Default: 100.',
                        type=float, default=100.)
    parser.add_argument('--gM_inh', help='Peak conductance of Mainen potassium ion channel for inhibitory neurons. Default: 0 (channel deactivated)',
                        type=float, default=0.)
    parser.add_argument('--ok_if_folder_exists', help='If set and the results folder already exists, previous results will be overwritten (mainly for testing)',
                        action='store_true', default=False)
    parser.add_argument('--rate_tasks', help='If set, the firing rate tasks are processed instead of the spike pattern classification tasks.',
                        action='store_true', default=False)
    parser.add_argument('--input_dimensions', help='Number of spike trains per input stream. Default for spike pattern '
                                                   'classification is 40 and for rate tasks is 4', type=int, default=None)
    parser.add_argument('--freeze_last_input', help='If set, the spike patterns of the last step for each trial is fixed',
                        action='store_true', default=False)
    parser.add_argument('--start_s2', help='Start time of input stream 2 in ms. Default value is 0 and should not be changed,'
                                           'because this messes up the task results. This is only needed to get a raster'
                                           'plot like in the Häusler Maass 2006 paper.', type=float, default=0.)
    parser.add_argument('--group_name', help='Name of the group of experiments. This is also the parent folder name '
                                             'inside the data folder, where the results are stored.',
                        default='defaultgroupname')
    parser.add_argument('--neuron_model', help='Neuron model (hh_cond_exp_destexhe or iaf_cond_exp)', default='hh_cond_exp_destexhe')
    parser.add_argument('--disable_conductance_noise', help='Disables the conductance noise in the HH neuron',
                        action='store_true', default=False)
    # parser.add_argument('--static_scaling', help='', type=float, default=73.)
    parser.add_argument('--vt_l23exc', help="Firing threshold for population L23 exc.", type=float, default=-52.)
    parser.add_argument('--vt_l23inh', help="Firing threshold for population L23 inh.", type=float, default=-55.)
    parser.add_argument('--vt_l4exc', help="Firing threshold for population L4 exc.", type=float, default=-49.)
    parser.add_argument('--vt_l4inh', help="Firing threshold for population L4 inh.", type=float, default=-55.)
    parser.add_argument('--vt_l5exc', help="Firing threshold for population L5 exc.", type=float, default=-57.)
    parser.add_argument('--vt_l5inh', help="Firing threshold for population L5 inh.", type=float, default=-65.)

    args = parser.parse_args()
    if args.S_rw is None:
        args.S_rw = 66825 / args.N
        if args.network_name == 'microcircuit_static':
            args.S_rw = args.S_rw / 73.

    if args.input_dimensions is None:
        if args.rate_tasks:
            args.input_dimensions = 4
        else:
            args.input_dimensions = 40

    if args.S1 is None:
        if args.input_dimensions == 4:
            args.S1 = 148.5
        else:
            args.S1 = 14.85
    if args.S2 is None:
        if args.input_dimensions == 4:
            args.S2 = 364.98
        else:
            args.S2 = 36.498

    return args


if __name__ == "__main__":
    arguments = parse_cmd()
    task_runner = SpikePatternTaskRunner(**vars(arguments))
    task_runner.run()
