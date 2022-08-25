import numpy as np
import nest

from experiments.tasks.base_tasks import BaseTasks


class FiringRateTask(BaseTasks):
    """ Class which defines the tasks that are based on the firing rates of the two input streams """

    def __init__(self, steps_per_trial=15, step_duration=30., discard_steps=0, train_trials=1500, test_trials=300,
                 minrate=15., maxrate=25., dimensions=4, rate_window=15., max_delay=0):
        super().__init__(steps_per_trial, step_duration, discard_steps, train_trials, test_trials, dimensions)

        self.rate_window = rate_window
        self.max_delay = max_delay

        # Input stream 1
        self.generator_target_rates_s1 = np.random.uniform(low=minrate, high=maxrate, size=self.total_num_steps)
        self.spike_generators_s1, self.spiketimes_per_generator_s1, self.input_values_s1 = self.create_spike_generators(self.generator_target_rates_s1)

        # Input stream 2
        self.generator_target_rates_s2 = np.random.uniform(low=minrate, high=maxrate, size=self.total_num_steps)
        self.spike_generators_s2, self.spiketimes_per_generator_s2, self.input_values_s2 = self.create_spike_generators(self.generator_target_rates_s2)

        self.targets = self.create_targets()

    def create_targets(self):
        """ Creates the task target values

        Returns
        -------
        dict
            dictionary with the target values for the different tasks

        """

        r1_array = np.array(self.get_standard_target_s1(delay=0))
        r2_array = np.array(self.get_standard_target_s2(delay=0))
        targets = {
            'r1/r2': r1_array/r2_array,
            '(r1-r2)^2': (r1_array - r2_array)**2,
        }
        if self.max_delay > 0:
            for delay in range(1, self.max_delay+1):
                delayed_r1_array = np.array(self.get_standard_target_s1(delay=delay))
                delayed_r2_array = np.array(self.get_standard_target_s2(delay=delay))
                targets[f'r1/r2_delay{delay}'] = delayed_r1_array / delayed_r2_array
                targets[f'(r1-r2)^2_delay{delay}'] = (delayed_r1_array - delayed_r2_array) ** 2

        return targets

    def create_spike_generators(self, input_values):
        """ Creates the devices that generate the spikes for the task inputs

        Parameters
        ----------
        input_values: ndarray or list
            input values which should be represented by the spike inputs

        Returns
        -------
        NEST devices
            NEST spike generators
        list
            list of list with the spike times for each generator
        list
            list of rates of the spike patterns

        """

        spike_generators = nest.Create('spike_generator', n=self.dimensions)
        spike_times_per_generator = dict((id, []) for id in spike_generators.global_id)

        def get_real_rate_in_window(spike_lists, end_time, window):
            dimensions = len(spike_lists)
            n_spikes_in_window = 0
            for spikes in spike_lists:
                n_spikes_in_window += sum(i > end_time - window for i in spikes)
            spkrate = 1000 * n_spikes_in_window / (window * dimensions)

            return spkrate

        real_rates = []

        for step, rate in enumerate(input_values):
            start = 0.1 + step * self.step_duration
            end = start + self.step_duration

            step_rate = 0.
            while step_rate == 0.:  # make sure that at least one spike is in the last rate_window ms (otherwise division by 0 in tasks)
                step_spike_lists = self.generate_single_spike_pattern(start_time=start, end_time=end, spikerate=rate, min_total_spikes=1)
                step_rate = get_real_rate_in_window(step_spike_lists, end, self.rate_window)

            for gen_id, generator_step_spikes in zip(spike_generators.global_id, step_spike_lists):
                spike_times_per_generator[gen_id].extend(generator_step_spikes)

            real_rates.append(step_rate)

        for spkgenerator, spike_times in zip(spike_generators, spike_times_per_generator.values()):
            nest.SetStatus(spkgenerator, {'spike_times': spike_times})

        return spike_generators, spike_times_per_generator, real_rates


if __name__ == "__main__":
    firingratetask = FiringRateTask()
