import copy
import warnings

import numpy as np
from pandas import DataFrame

from . import extractors, readouts
from fna import tools
from fna.tools import parameters as par

logger = tools.utils.logger.get_logger(__name__)


class Decoder(object):
    """
    Attached to a Network object, a Decoder object is a container for all the readouts, independent of
    the underlying network type (spiking, ESN, etc.).
    """

    def __init__(self, decoder_parameters, rng=None):
        """
        Creates Decoder object and sanitizes the initializer, but does not yet create the Readouts
         nor does it connect anything to the network (and state extractor).

        :param decoder_parameters: ParameterSet object or dictionary specifying decoding parameters
        """
        self.decoding_pars = self._parse_parameters(decoder_parameters)

        # when processing the first batch (e.g., training), the decoder is initialized with the targets and tasks,
        # creating the task-specific readouts and decoder helper containers.
        self.initialized = False
        self.target_outputs = None
        # self.readouts = {}  # dictionary with {readout_label: {extractor_label: {task: ... } } }
        self.readouts = []  # list of all Readout objects
        self.validation_accuracy = {}

        if rng is None:
            self.rng = np.random.default_rng()
            logger.warning("Decoder results may not be reproducible!")
        else:
            self.rng = rng

    @staticmethod
    def _parse_parameters(initializer):
        """

        :param initializer:
        :return:
        """
        if isinstance(initializer, dict):
            initializer = par.ParameterSet(initializer)
        assert isinstance(initializer, par.ParameterSet), "Decoder must be initialized with ParameterSet or dictionary"
        return initializer

    @staticmethod
    def _parse_target_outputs(target_outputs):
        """
        Consistency check for the target outputs list.

        :param target_outputs: [list] list of dictionaries specifying the task=specific target outputs for the readouts
        :return: [dict] target_outputs { label: { output: [], accept: [] } }
        """
        assert isinstance(target_outputs, list), 'Target outputs in DecodingLayer must be of type list!'
        required_keys = ['label', 'output', 'accept']
        for item in target_outputs:
            par.validate_keys(item, required_keys)

        return {item['label']: {'output': item['output'], 'accept': item['accept']} for item in target_outputs}

    def connect(self, state_extractor, target_outputs):
        """
        Processes the decoding parameters, creating the required Readout objects for each state matrix / extractor -
        task combination.

        :param state_extractor: [list] list of strings with the state extractor labels
        :param target_outputs:
        :return:
        """
        logger.info('Initializing and connecting decoder...')
        # assert isinstance(state_extractor, extractors.SpikingExtractor)

        target_outputs = self._parse_target_outputs(target_outputs)
        task_labels = target_outputs.keys()
        if isinstance(state_extractor, extractors.SpikingExtractor):
            all_extractor_labels = state_extractor.get_labels()
        else:
            all_extractor_labels = state_extractor

        logger.info('Creating readouts:')
        # iterate through all the readout parameters in the dictionary
        for r_label, r_params in self.decoding_pars.items():
            # get all the extractor - task combinations which require a readout
            if 'extractor' in r_params:
                if r_params.extractor in all_extractor_labels:
                    decode_extractor_labels = [r_params.extractor]
                else:
                    warnings.warn('No matching extractor `{}` for readout `{}`'.format(r_params.extractor, r_label))
                    continue
            else:
                decode_extractor_labels = all_extractor_labels

            # process each extractor - task combinations and create a corresponding readout
            for e_l in decode_extractor_labels:
                # update with correct extractor label
                tmp_r_params = copy.deepcopy(r_params)
                tmp_r_params['extractor'] = e_l

                for t_l in task_labels:
                    tmp_r_params['task'] = t_l
                    self.readouts.append(readouts.Readout(r_label, tmp_r_params, rng=self.rng))

        self.initialized = True

    @staticmethod
    def _unfold_targets_to_samples(stim_info, sampled_times_stim_tz, task_targets, accepted):
        """
        Maps / expands the target outputs onto the sampled states. Whereas the target outputs have one value for
        each stimulus, different sampling strategies can yield multiple samples per stimulus.

        TODO this assumes that task_targets is a 1-dimensional array
        :param stim_info:
        :param sampled_times_stim_tz:
        :param task_targets:
        :param accepted:
        :return: (np.array, np.array) - unfolded targets and accepted arrays
        """
        res_targets = []
        res_accepted = []

        assert len(stim_info) == len(task_targets) == len(accepted)
        total_samples = 0
        for idx in range(len(stim_info)):
            t_start, t_stop, interval = stim_info[idx]
            # number of samples collected during the current (idx) stimulus
            n_stim_samples = len(np.where((sampled_times_stim_tz + 0.0001 > t_start )
                                          & (sampled_times_stim_tz + 0.0001 < t_stop))[0])
            total_samples += n_stim_samples
            # unfold targets and accepted arrays
            res_targets.append(np.repeat(task_targets[idx], n_stim_samples))
            res_accepted.append(np.repeat(accepted[idx], n_stim_samples))

        return np.concatenate(res_targets), np.concatenate(res_accepted)

    def train(self, state_extractor, batch_label, batch_target_outputs, stim_info, vocabulary=None):
        """

        :param state_extractor:
        :param batch_target_outputs:
        :param batch_label:
        :param stim_info:
        :param vocabulary: TODO rewrite!! readout output dimension, which may differ from the target dimension if some elements
                           are not seen during training (in some cases it may correspond to the vocabulary size)
        :return:
        """
        target_outputs = self._parse_target_outputs(batch_target_outputs)

        for readout in self.readouts:
            logger.info('Processing readout [{}-{}] with state [{}]'.format(readout.label, readout.task, readout.extractor))

            if isinstance(state_extractor, extractors.SpikingExtractor):
                state = state_extractor.get_state_matrix(readout.extractor)
            else:
                state = state_extractor[readout.extractor]

            state_matrix = state.matrix

            if isinstance(state_extractor, extractors.SpikingExtractor):
                # network's timezone (includes encoding delay_
                sampled_times = state.sampled_times[0]
                # stimulus' timezone (shifted to stimulus onset, discard encoding delay)
                sampled_times_stim_tz = np.array(sampled_times) - state_extractor.encoder_delay
            else:
                sampled_times = state.sampled_times
                sampled_times_stim_tz = np.array(sampled_times)

            # unfold targets to samples
            task_targets = np.array(target_outputs[readout.task]['output'])
            accepted = np.array(target_outputs[readout.task]['accept'])

            if isinstance(stim_info[0], tuple):
                unfolded_task_targets, unfolded_accepted = \
                    self._unfold_targets_to_samples(stim_info, sampled_times_stim_tz, task_targets, accepted)
                # train readout
                readout.train(batch_label, state_matrix, unfolded_task_targets, accepted=unfolded_accepted,
                              vocabulary=vocabulary)
            else:
                readout.train(batch_label, state_matrix, task_targets, accepted=accepted)

    def validate(self, batch_label, symbolic=False, vocabulary=None):
        """
        Evaluates the performance (error) of the current readout state. When called after processing the last batch
        within an epoch, it provides the loss (MSE) in order to evaluate the quality of training.
        Calling this function assumes the readout outputs are available (have been generated).

        :return: [dict] dictionary with performance data
        """
        self.validation_accuracy.update({batch_label: self.evaluate(process_output_method=None, vocabulary=vocabulary,
                                                                    flush=True, symbolic=symbolic, mse_only=True)})
        return self.validation_accuracy

    def predict(self, state_extractor, batch_target_outputs, stim_info, vocabulary=None):
        """
        Compute readout predictions.

        :param state_extractor: [SpikingExtractor or dict]
        :param batch_target_outputs:
        :param stim_info: [list]
        :return:
        """
        target_outputs = self._parse_target_outputs(batch_target_outputs)

        for readout in self.readouts:
            if readout.task in target_outputs.keys():
                logger.info('Processing readout [{}] with state [{}]'.format(readout.label, readout.extractor))
                if isinstance(state_extractor, extractors.SpikingExtractor):
                    state = state_extractor.get_state_matrix(readout.extractor)
                else:
                    state = state_extractor[readout.extractor]

                state_matrix = state.matrix

                if isinstance(state_extractor, extractors.SpikingExtractor):
                    # network's timezone (includes encoding delay_
                    sampled_times = state.sampled_times[0]
                    # stimulus' timezone (shifted to stimulus onset, discard encoding delay)
                    sampled_times_stim_tz = np.array(sampled_times) - state_extractor.encoder_delay
                else:
                    sampled_times = state.sampled_times
                    sampled_times_stim_tz = np.array(sampled_times)

                # unfold targets to samples
                task_targets = np.array(target_outputs[readout.task]['output'])
                accepted = np.array(target_outputs[readout.task]['accept'])

                if isinstance(stim_info[0], tuple):
                    unfolded_task_targets, unfolded_accepted = \
                        self._unfold_targets_to_samples(stim_info, sampled_times_stim_tz, task_targets, accepted)

                    readout.predict(state_matrix, unfolded_task_targets, accepted=unfolded_accepted, vocabulary=vocabulary)
                else:
                    readout.predict(state_matrix, task_targets, accepted=accepted)

    def evaluate(self, process_output_method=None, symbolic=True, flush=False, mse_only=True, vocabulary=None):
        """

        :param process_output_method: [str] 'k-WTA' or ...
        :param symbolic: [bool] symbolic sequence or not
        :param flush: [bool]
        :param mse_only:
        :return:
        """
        perf_dict = {}
        for readout in self.readouts:
            if readout.output is not None and readout.test_target is not None:
                r_perf = readout.evaluate(process_output_method=process_output_method, symbolic=symbolic,
                                          mse_only=mse_only, vocabulary=vocabulary)
                perf_dict.update({readout.label+'-'+readout.task: r_perf})
                if flush:
                    readout.performance = {}
        df_dict = {}
        for k, v in perf_dict.items():
            df_dict.update({k: {}})
            for k1, v1 in v.items():
                if k1 in ['raw', 'max', 'label']:
                    for k2, v2 in v1.items():
                        df_dict[k].update({k1+'-'+k2: v2})
        perf_data = DataFrame(df_dict)
        logger.info("Decoding performance: \n{0}".format(perf_data))
        return perf_data

    def save_training_data(self, label=None):
        """
        Saves readout data, such as OutputMapper.
        :return:
        """
        for readout in self.readouts:
            readout.weights.save(label)

    def retrieve_outputs(self):
        """
        Retrieves the last outputs and targets from all readouts
        """
        outputs = [x.output for x in self.readouts]
        targets = [x.test_target for x in self.readouts]
        return outputs, targets