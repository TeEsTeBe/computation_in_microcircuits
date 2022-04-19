import numpy as np
import nest

from experiments.tasks.base_tasks import BaseTasks


class SpikePatternClassification(BaseTasks):

    def __init__(self, steps_per_trial=15, step_duration=30., discard_steps=0, train_trials=1500, test_trials=300,
                 spikerate=20., dimensions=40, templates_per_step=2, jitter_std=1., freeze_last_input=False, start_s2=0.,
                 max_delay=1):
        super().__init__(steps_per_trial, step_duration, discard_steps, train_trials, test_trials, dimensions)

        self.spikerate = spikerate
        self.templates_per_step = templates_per_step
        self.jitter_std = jitter_std
        self.freeze_last_input = freeze_last_input
        self.start_s2 = start_s2
        self.max_delay = max_delay

        # Input stream 1
        self.input_values_s1 = np.random.randint(0, 2, size=self.total_num_steps)
        if self.freeze_last_input:
            self.input_values_s1[self.steps_per_trial-1::self.steps_per_trial] = np.zeros(self.train_trials+self.test_trials)

        self.spike_pattern_templates_per_step_S1 = self.generate_spike_pattern_templates()
        self.spike_generators_s1, self.spiketimes_per_generator_s1 = self.create_spike_generators(
            self.input_values_s1,
            self.spike_pattern_templates_per_step_S1
        )

        # Input stream 2
        self.input_values_s2 = np.random.randint(0, 2, size=self.total_num_steps)
        if self.freeze_last_input:
            self.input_values_s2[self.steps_per_trial-1::self.steps_per_trial] = np.zeros(self.train_trials+self.test_trials)

        self.spike_pattern_templates_per_step_S2 = self.generate_spike_pattern_templates()
        self.spike_generators_s2, self.spiketimes_per_generator_s2 = self.create_spike_generators(
            self.input_values_s2,
            self.spike_pattern_templates_per_step_S1,
            start=self.start_s2
        )

        self.targets = self.create_targets()

    def create_targets(self):
        targets = {
            'spike_pattern_classification_S1': self.get_standard_target_s1(delay=0),
            'spike_pattern_classification_S2': self.get_standard_target_s2(delay=0),
            'delayed_spike_pattern_classification_S1': self.get_standard_target_s1(delay=1),
            'delayed_spike_pattern_classification_S2': self.get_standard_target_s2(delay=1),
            'xor_spike_pattern': self.get_xor_target(delay=0),
            'xor_spike_pattern_delay1': self.get_xor_target(delay=1),
        }

        if self.max_delay > 1:
            for delay in range(2, self.max_delay+1):
                targets[f'delayed_spike_pattern_classification_S1_delay{delay}'] = self.get_standard_target_s1(delay=delay)
                targets[f'delayed_spike_pattern_classification_S2_delay{delay}'] = self.get_standard_target_s2(delay=delay)
                targets[f'xor_spike_pattern_delay{delay}'] = self.get_xor_target(delay=delay)

        return targets

    def generate_spike_pattern_templates(self):
        # TODO: docstring
        spike_pattern_templates = [[] for _ in range(self.steps_per_trial)]
        for step_nr in range(self.steps_per_trial):
            start_time = 0.1
            end_time = self.step_duration
            for _ in range(self.templates_per_step):
                spike_pattern_templates[step_nr].append(
                    self.generate_single_spike_pattern(start_time=start_time, end_time=end_time, spikerate=self.spikerate)
                )

        return spike_pattern_templates

    def add_jitter(self, spike_times, max_time, min_time=0.1):

        if len(spike_times) > 0:
            noise_array = np.random.normal(0, scale=self.jitter_std, size=len(spike_times))
            jittered_spikes = spike_times + noise_array

            # Confine spikes to be inside the min_time, max_time boundary
            lower_bound = min(min_time, jittered_spikes.min())
            upper_bound = max(max_time, jittered_spikes.max())
            jittered_spikes = np.interp(jittered_spikes, (lower_bound, upper_bound), (min_time, max_time))

            jittered_spikes = np.around(jittered_spikes, decimals=1)  # spike times need to be on the NEST time grid
            jittered_spikes = np.sort(jittered_spikes)  # noise can lead to switching of positions -> need to be ordered for NEST
        else:
            jittered_spikes = spike_times

        return jittered_spikes

    def create_spike_generators(self, input_values, spike_templates_per_step, start=0.):
        n_generators = len(spike_templates_per_step[0][0])
        spike_generators = nest.Create('spike_generator', n=n_generators)
        spike_times_per_generator = dict((id, []) for id in spike_generators.global_id)

        n_substeps = len(spike_templates_per_step)

        for input_nr, val in enumerate(input_values):
            step_nr = input_nr % n_substeps
            is_last_step_of_trial = step_nr == self.steps_per_trial - 1
            jitter_this_step = not self.freeze_last_input or not is_last_step_of_trial
            pattern = spike_templates_per_step[step_nr][val]
            for generator_id, spike_times in zip(spike_times_per_generator.keys(), pattern):
                spktms = np.array(spike_times).copy()
                if jitter_this_step:
                    spktms = self.add_jitter(spktms, min_time=0.1, max_time=self.step_duration)
                spike_times = spktms + input_nr * self.step_duration
                spike_times_per_generator[generator_id].extend(spike_times)

        for spkgenerator, spike_times in zip(spike_generators, spike_times_per_generator.values()):
            nest.SetStatus(spkgenerator, {'spike_times': spike_times, 'start': start})

        return spike_generators, spike_times_per_generator

    def get_xor_target(self, delay=0):
        s1_target = self.get_standard_target_s1(delay=delay)
        s2_target = self.get_standard_target_s2(delay=delay)

        xor_target = np.zeros_like(s1_target)
        xor_target[np.logical_xor(s1_target, s2_target)] = 1.

        return xor_target
