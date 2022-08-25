from abc import ABC
import numpy as np


class BaseTasks(ABC):
    """ Abstract base class that defines all functions and variables that need to be implemented by a task """

    def __init__(self, steps_per_trial, step_duration, discard_steps, train_trials, test_trials, dimensions):
        self.steps_per_trial = steps_per_trial
        self.step_duration = step_duration
        self.discard_steps = discard_steps
        self.train_trials = train_trials
        self.test_trials = test_trials
        self.total_num_steps = steps_per_trial * (train_trials + test_trials) + discard_steps
        self.simtime = self.total_num_steps * step_duration + 0.1
        self.dimensions = dimensions

        self.targets = None
        self.input_values_s1 = None
        self.input_values_s2 = None
        self.spike_times_per_generator_s1 = None
        self.spike_times_per_generator_s2 = None
        self.spike_generators_s1 = None
        self.spike_generators_s2 = None

    def generate_single_spike_pattern(self, start_time, end_time, spikerate, min_total_spikes=0):
        """ Generates a spike pattern for a single step of the input for the task

        Parameters
        ----------
        start_time: float
            earliest possible time for the first spike in the pattern
        end_time: float
            latest possible time for the last spike in the pattern
        spikerate: float
            target firing rate for the spike pattern
        min_total_spikes: int
            minimum number of spikes that have to be part of the spike pattern

        Returns
        -------
        list
            list of lists which contain the spike times of the spike pattern

        """

        duration = end_time - start_time + 0.1
        spike_times_lists = [[] for _ in range(self.dimensions)]

        total_num_spikes = self.dimensions * duration / 1000. * spikerate  # for all generators/dimensions
        add_spike_probability = total_num_spikes - int(total_num_spikes)
        total_num_spikes = int(total_num_spikes)
        if np.random.uniform(0, 1) < add_spike_probability:
            total_num_spikes += 1

        total_num_spikes = max(total_num_spikes, min_total_spikes)

        all_spike_times = sorted(np.random.uniform(low=start_time + 0., high=end_time, size=total_num_spikes))
        all_spike_times = [round(spkt, 1) for spkt in all_spike_times]

        for i, spike_time in enumerate(all_spike_times):
            target_dim = np.random.randint(0, self.dimensions)
            spike_times_lists[target_dim].append(spike_time)

        return spike_times_lists

    def get_standard_target_s1(self, delay=0):
        """ Returns the standard target for the task based on the first input stream

        Parameters
        ----------
        delay: int
            number of delays that should be taken into account

        Returns
        -------
        ndarray
            numpy array with target values

        """

        return self.input_values_s1[self.discard_steps + self.steps_per_trial - delay - 1::self.steps_per_trial]

    def get_standard_target_s2(self, delay=0):
        """ Returns the standard target for the task based on the second input stream

        Parameters
        ----------
        delay: int
            number of delays that should be taken into account

        Returns
        -------
        ndarray
            numpy array with target values

        """

        return self.input_values_s2[self.discard_steps + self.steps_per_trial - delay - 1::self.steps_per_trial]
