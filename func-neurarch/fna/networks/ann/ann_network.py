import tensorflow as tf
import numpy as np
import os

from fna.decoders import Decoder, StateMatrix

from fna.tools.network_architect import Network
from fna.tools.network_architect.connectivity import TFConnector
from fna.tools.parameters import ParameterSet
from fna.tools import utils

logger = utils.logger.get_logger(__name__)


class ArtificialNeuralNetwork(Network):
    """
    Wrapper for ANNs, inheriting the properties of generic Network object and extending
    them with specific methods
    """

    def __init__(self, network_parameters, input_dim, output_dim, extractors=None, decoders=None,
                 label=None):
        """
        ANN instance constructor.

        :param network_parameters: dictionary
        :param input_dim: [int] input dimensions
        :param output_dim: [int] output dimensions
        :param extractors: None or dict
        :param decoders: None or dict

        - Implemented architectures: Vanilla, UG-RNN, GRU, LSTM
        - Activation functions: tanh, relu, sigmoid
        """
        self.name = label
        self.simulator = 'TensorFlow'
        logger.info("Initializing {0!s} architecture ({1!s}-simulated)".format(self.name, self.simulator))
        self._parse_parameters(network_parameters)

        self._n_input = input_dim
        self._n_output = output_dim
        self.gids = np.arange(self.n_neurons).astype(int)

        self.variables = {}
        self.states = {}
        self._sess = None
        self._graph = 0
        self._n_batches = 0
        self.tf_data = {}
        self.decoders = None
        self._global_step = None

        self.cell = self.create_population(self._cell_type, N=self.n_neurons, trf=self._trf)

        if extractors is not None:
            self._state_extractors = extractors
            self._extractor_labels = list(extractors.keys())
        if decoders is not None:
            self.decoders = Decoder(decoders)

        self.report()

    def _parse_parameters(self, parameters):
        """
        Unpack a compact set of parameters, from which all relevant info is extracted
        :return:
        """
        if isinstance(parameters, dict):
            parameters = ParameterSet(parameters)
        self.n_neurons = parameters.N
        self._cell_type = parameters.cell_type
        self._learning_rate = parameters.learning_rate
        if hasattr(parameters, "l2_loss"):
            self._loss = parameters.l2_loss
        else:
            self._loss = 0.01
        if hasattr(parameters, "optimizer"):
            self._opt = parameters.optimizer
        else:
            self._opt = "momentum"
        if hasattr(parameters, "transfer_function"):
            self._trf = parameters.transfer_function
        else:
            self._trf = "tanh"

    @staticmethod
    def create_population(cell_type, N, trf):
        logger.info("Creating populations:")
        logger.info("- Population {0!s} [N={1!s}]".format(cell_type, N))

        if cell_type == 'Vanilla':
            if trf == 'tanh':
                cell = tf.contrib.rnn.BasicRNNCell(N, reuse=tf.AUTO_REUSE, activation=tf.nn.tanh)
            elif trf == 'relu':
                cell = tf.contrib.rnn.BasicRNNCell(N, reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
            else:
                raise NotImplementedError("{0!s} transfer function not currently supported".format(trf))

        elif cell_type == 'GRU':
            if trf == 'tanh':
                cell = tf.contrib.rnn.GRUCell(N, reuse=tf.AUTO_REUSE, activation=tf.nn.tanh)
            elif trf == 'relu':
                cell = tf.contrib.rnn.GRUCell(N, reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
            else:
                raise NotImplementedError("{0!s} transfer function not currently supported".format(trf))

        elif cell_type == 'LSTM': # TODO - check state tuple options (BasicLSTMCell)
            if trf == 'tanh':
                cell = tf.contrib.rnn.LSTMCell(N, reuse=tf.AUTO_REUSE, state_is_tuple=True, activation=tf.nn.tanh)
            elif trf == 'relu':
                cell = tf.contrib.rnn.LSTMCell(N, reuse=tf.AUTO_REUSE, state_is_tuple=True, activation=tf.nn.relu)
            else:
                raise NotImplementedError("{0!s} transfer function not currently supported".format(trf))

        elif cell_type == 'UGRNN':
            if trf == 'tanh':
                cell = tf.contrib.rnn.UGRNNCell(N, reuse=tf.AUTO_REUSE, activation=tf.nn.tanh)
            elif trf == 'relu':
                cell = tf.contrib.rnn.UGRNNCell(N, reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
            else:
                raise NotImplementedError("{0!s} transfer function not currently supported".format(trf))

        else:
            raise NotImplementedError("{0!s} cell not implemented".format(cell_type))

        return cell

    def initialize_states(self):
        """
        Initialize state variables
        :return:
        """
        init_state = self.cell.zero_state(self._n_batches, dtype=tf.float32)
        self.tf_data.update({'init_state': init_state})

    def clone(self):
        pass

    def report(self):
        """
        Print a description of the system
        """
        logger.info("========================================================")
        logger.info(" {0!s} architecture ({1!s}-simulated):".format(self.name, self.simulator))
        logger.info("--------------------------------------------------------")
        logger.info("- Size: {0!s}".format(self.n_neurons))
        logger.info("- Unit: {0!s} [{1!s} activation]".format(self._cell_type, self._trf))
        logger.info("- Hyperparameters: \n\t- learning rate = {0!s}\n\t- L2 loss = {1!s}\n\t- Optimizer: {2!s}".format(
            self._learning_rate, self._loss, self._opt))

    def reset_graph(self):
        if self._sess:
            self._sess.close()
        tf.reset_default_graph()

    def prepare_for_batch_processing(self, n_batches=1, batch_size=100):
        """
        Setup placeholders for the input and output tensors
        :param n_batches:
        :param batch_size: currently needs to be fixed
        :return:
        """
        u = tf.placeholder(tf.float32, [n_batches, batch_size, self._n_input], name='input_placeholder')
        z = tf.placeholder(tf.float32, [n_batches, batch_size, self._n_output], name='output_placeholder')
        self.tf_data.update({'input': u, 'output': z})
        return u, z

    def get_states(self, state, input_times=None, dataset_label=None):
        """
        Store population states
        :param batch_state:
        :param input_times:
        :param dataset_label:
        :param save:
        :return:
        """
        sampled_times = input_times
        for ext_label, ext_params in self._state_extractors.items():
            self.states.update({ext_label: StateMatrix(state, ext_label, state_var=ext_params['variable'],
                                                           population=self.name, sampled_times=sampled_times,
                                                           dataset_label=dataset_label, save=ext_params['save'])})
            if "standardize" in ext_params.keys():
                if ext_params['standardize']:
                    self.states[ext_label].standardize()

    def setup_optimizer(self, loss):
        if self._opt == 'adam':
            optimizer = tf.train.AdagradOptimizer(self._learning_rate)
            opt = optimizer.minimize(loss, global_step=self._global_step)
        elif self._opt == 'momentum':
            decay = tf.train.exponential_decay(self._learning_rate, self._global_step, 1, 0.9)
            optimizer = tf.train.MomentumOptimizer(decay, 0.5)
            gradients, variables = zip(*optimizer.compute_gradients(loss, tf.trainable_variables()))
            gradients = [None if gradient is None else tf.clip_by_norm(gradient, 2.0) for gradient in gradients]
            opt = optimizer.apply_gradients(zip(gradients, variables), global_step=self._global_step)
        else:
            raise NotImplementedError("Optimizer type {0!s} not currently implemented".format(self._opt))
        return opt

    def build_graph(self, input_placeholder, output_placeholder):
        # self.reset_graph()
        self.initialize_states()
        rnn_outputs, final_state = tf.nn.dynamic_rnn(self.cell, input_placeholder, initial_state=self.tf_data[
            'init_state'])

        scale = 1.0 / np.sqrt(self.n_neurons)
        W = np.multiply(scale, np.random.randn(self.n_neurons, self._n_output))
        b = np.zeros(self._n_output)

        with tf.variable_scope('losses',reuse=tf.AUTO_REUSE):
            W = tf.Variable(W, dtype=tf.float32)
            b = tf.Variable(b, dtype=tf.float32)

        # with tf.variable_scope('losses', reuse=tf.AUTO_REUSE):
        #     W = tf.get_variable('W', [self.n_neurons, self._n_output])
        #     b = tf.get_variable('b', [self._n_output], initializer=tf.constant_initializer(0.0))

        rnn_outputs_ = tf.reshape(rnn_outputs, [-1, self.n_neurons])
        logits = tf.tensordot(rnn_outputs_, W, axes=1) + b

        self._global_step = tf.train.get_or_create_global_step()
        vars_ = tf.trainable_variables()

        y_as_list = tf.reshape(output_placeholder, [-1, self._n_output])

        loss = tf.nn.l2_loss(W) * self._loss
        losses = tf.squared_difference(logits, y_as_list)

        total_loss = tf.reduce_mean(losses)
        train_step = self.setup_optimizer(total_loss)

        # train_step = tf.train.AdagradOptimizer(self._learning_rate).minimize(total_loss)

        self._graph = 1
        self.tf_data.update({'losses': total_loss, 'train_step': train_step, 'hiddens': rnn_outputs,
                             'finalstate': final_state, 'predict': logits, 'saver': tf.train.Saver()})

    def train(self, data_batch, n_epochs=1, verbose=True, save=False, gpu_id=None, symbolic=True, label=''):
        """
        Supervised training
        :return:
        """
        in_ph, out_ph = self.prepare_for_batch_processing(n_batches=len(data_batch['inputs']), batch_size=data_batch[
            'inputs'][0].shape[0])
        self._n_batches = len(data_batch['inputs'])
        self.build_graph(in_ph, out_ph)

        saver = tf.train.Saver()
        training_losses = []
        states = []
        outputs = []
        dec_outputs = []
        dec_targets = []
        dec_loss = []
        perf = None
        decoder_targets = data_batch['decoder_outputs']

        with tf.Session(config=tf.ConfigProto()) as self._sess:
            device = '/cpu:0' if gpu_id is None else '/gpu:0'
            logger.info("Running TF session on {}".format(device))

            self._sess.run(tf.global_variables_initializer())

            for epoch in range(n_epochs):
                evaluate = [self.tf_data['losses'], self.tf_data['train_step'], self.tf_data['finalstate'],
                            self.tf_data['hiddens'], self.tf_data['predict']]

                (tr_losses, training_step_, f_state, state, output) = self.process_batch(evaluate, data_batch)

                if self.decoders is not None:
                    for n_batch in range(self._n_batches):
                        batch_label = "train_batch={}_epoch={}".format(n_batch+1, epoch+1)
                        st = state[n_batch, :, :].T
                        self.get_states(st, input_times=data_batch['input_times'], dataset_label=batch_label)

                        if not self.decoders.initialized:
                            self.decoders.connect(self._extractor_labels, decoder_targets[n_batch])

                        self.decoders.train(self.states, batch_label, decoder_targets[n_batch],
                                            stim_info=data_batch['input_times'])
                    self.decoders.validate(batch_label, symbolic)
                    if epoch == n_epochs - 1:
                        dec_loss = self.decoders.validation_accuracy
                        self.decoders.predict(self.states, decoder_targets[n_batch],
                                              stim_info=data_batch['input_times'])
                        perf = self.decoder_accuracy(output_parsing='k-WTA', symbolic_task=symbolic)

                if verbose:
                    logger.info("Epoch {0} loss: {1}".format(epoch+1, tr_losses))
                training_losses.append(tr_losses)

                if epoch == 0 or epoch == n_epochs-1:
                    states.append(state)
                    outputs.append(output)
                    if self.decoders is not None:
                        outs, tgts = self.decoders.retrieve_outputs()
                        dec_outputs.append(outs)
                        dec_targets.append(tgts)

            if save:
                pth = utils.data_handling.paths['other'] + label + '/'
                if not os.path.exists(pth):
                    os.makedirs(pth)
                saver.save(self._sess, pth)

                if self.decoders is not None:
                    self.decoders.save_training_data()
        # get trained parameter values

        return {'losses': training_losses, 'outputs': outputs, 'states': states, 'decoder_loss': dec_loss,
                'decoder_accuracy': perf, 'decoder_outputs': dec_outputs, 'decoder_targets': dec_targets}

    def test(self, data_batch, output_parsing='k-WTA', symbolic=True, verbose=True, save=False, gpu_id=None, label=''):
        """
        Process a test set
        :param data_batch:
        :param output_parsing:
        :param symbolic:
        :return:
        """
        test_results = self.predict(data_batch, verbose, gpu_id, save=save, label=label)
        perf = []
        decoder_outputs = data_batch['decoder_outputs']
        if self.decoders is not None:
            n_batch = -1
            batch_label = "test_batch={}".format(n_batch)
            st = test_results['states'][n_batch, :, :].T
            self.get_states(st, input_times=data_batch['input_times'], dataset_label=batch_label)
            self.decoders.predict(self.states, decoder_outputs[n_batch],
                                  stim_info=data_batch['input_times'])
            perf = self.decoder_accuracy(output_parsing=output_parsing, symbolic_task=symbolic)
            outs, tgts = self.decoders.retrieve_outputs()
            test_results.update({'decoder_outputs': outs, 'decoder_targets': tgts})
        test_results.update({'decoder_accuracy': perf})
        return test_results

    def predict(self, data_batch, verbose=True, gpu_id=None, save=False, label=''):
        """
        Runs the network without training (pass data batch and generate outputs)
        :return:
        """
        saver = tf.train.Saver()
        pth = utils.data_handling.paths['other'] + label + '/'
        chkpt_ = tf.train.get_checkpoint_state(pth)

        with tf.Session(config=tf.ConfigProto()) as self._sess:
            device = '/cpu:0' if gpu_id is None else '/gpu:0'
            logger.info("Running TF session on {}".format(device))

            self._sess.run(tf.global_variables_initializer())
            logger.info("Restoring from checkpoint")
            saver.restore(self._sess, chkpt_.model_checkpoint_path)

            evaluate = [self.tf_data['hiddens'], self.tf_data['predict']]
            (hidden_state, predict) = self.process_batch(evaluate, data_batch)

            if save:
                pth = utils.data_handling.paths['other']
                saver.save(self._sess, pth)

        return {'states': hidden_state, 'outputs': predict}

    def process_batch(self, ev_options, data_batch):
        """
        Parse one batch of input/output data
        :return:
        """
        session_outputs = self._sess.run(ev_options, feed_dict={self.tf_data['input']: np.array(data_batch['inputs']),
                                                                self.tf_data['output']: np.array(data_batch['targets'])})
        return session_outputs

    def decoder_accuracy(self, output_parsing='k-WTA', symbolic_task=True, store=True):
        """
        Evaluate the accuracy of the decoders on the test set
        """
        return self.decoders.evaluate(process_output_method=output_parsing, symbolic=symbolic_task,
                                      flush=not store, mse_only=not symbolic_task)


# ######################################################################################################################
class ContinuousRateRNN(Network):
    """
    Implement continuous rate model
    """
    def __init__(self, network_parameters, connection_parameters, input_dim, output_dim, dt=0.1, extractors=None,
                 decoders=None, label=None):
        """
        Instantiate model
        :return:
        """
        self.name = label
        self.simulator = "TensorFlow"
        logger.info("Initializing {0!s} architecture ({1!s}-simulated)".format(self.name, self.simulator))
        self._parse_parameters(network_parameters, parameter_type='network')

        self._n_input = input_dim
        self._n_output = output_dim
        self.gids = np.arange(self.n_neurons).astype(int)

        self._resolution = dt
        self.variables = {}
        self.states = {}
        self.dec_states = {}

        self._sess = None
        self._graph = 0
        self._n_batches = 0
        self.tf_data = {}
        self.connection_masks = {}
        self.decoders = None

        # generate connectivity - ConnectionMapper object / tf.Variable
        if self._EI_ratio:
            nE = self.n_neurons * self._EI_ratio
            ei_list = np.ones(self.n_neurons, dtype=int)
            ei_list[int(nE):] = -1
            self.gids *= ei_list # negative gids will correspond to inh neurons

        self.outputs = None
        self.connections = None
        self.w_rec = None

        self.reset_graph()
        self.build_graph()

        ei_balance = connection_parameters['w_rec']['ei_balance']
        self.connect(connection_parameters, ei_balance=ei_balance)

        if extractors is not None:
            self._state_extractors = extractors
            self._extractor_labels = list(extractors.keys())
        if decoders is not None:
            self.decoders = Decoder(decoders)

        self.report()

    def _parse_parameters(self, parameters, parameter_type='network'):
        """
        Unpack a compact set of parameters, from which all relevant info is extracted
        :return:
        """
        if isinstance(parameters, dict):
            parameters = ParameterSet(parameters)
        if parameter_type == 'network':
            self.n_neurons = parameters.N
            self._trf = parameters.transfer_fcn
            # self._rec_density = parameters.connection_density
            self._EI_ratio = parameters.EI_ratio
            self._taus = (parameters.tau_x, parameters.tau_r)
            self._noise = parameters.noise
            self._initialize_states = parameters.initial_state
            self._learning_rate = parameters.learning_rate
        elif parameter_type == 'connections':
            for k, v in parameters.items():
                if k[-2:] == 'in':
                    dim = [self._n_input, self.n_neurons]
                elif k[-3:] == 'rec':
                    dim = [self.n_neurons, self.n_neurons]
                elif k[-3:] == 'out':
                    dim = [self.n_neurons, self._n_output]
                else:
                    raise ValueError("{} is not an accepted connection parameter")

                if isinstance(v, dict):
                    parameters[k].update({'dimensions': dim})
                elif isinstance(v, np.ndarray):
                    assert np.shape(v) == (self._n_input, self.n_neurons), "Incorrect shape for weight matrix {}".format(k)
                else:
                    raise NotImplementedError("Weight matrices as {} cannot be used".format(str(type(v))))
            return parameters

    def report(self):
        """
        Print a description of the system
        """
        # TODO - extend (see SNN)
        logger.info("========================================================")
        logger.info(" {0!s} architecture ({1!s}-simulated):".format(self.name, self.simulator))
        logger.info("--------------------------------------------------------")
        logger.info("- Size: {0!s}".format(self.n_neurons))
        logger.info("- Neuron models: {0!s}".format(self._trf))

    def initialize_states(self):
        """
        Set the initial state
        :return:
        """
        logger.info("Initializing state variables:")
        if self._initialize_states is None:
            return 0.1 * np.ones((1, self.n_neurons), dtype=np.float32)
        else:
            assert isinstance(self._initialize_states, tuple), "Initial states must be a tuple or None (for default " \
                                                               "initialization)"
            return self._initialize_states[0](size=(1, self.n_neurons), **self._initialize_states[1]).astype(np.float32)

    def connect(self, connection_pars, ei_balance=1.):
        """
        Setup connection weights
        :param connection_pars: dict - must contain "w_in", "w_rec" and "w_out"
        :param ei_balance: if connections are split into E/I, how much stronger should I be?
        """
        connection_pars = self._parse_parameters(connection_pars, parameter_type="connections")
        self.connections = TFConnector(self, connection_pars)

        w_in_mask = np.ones((self._n_input, self.n_neurons), dtype=np.float32)
        w_rec_mask = np.ones((self.n_neurons, self.n_neurons), dtype=np.float32)
        w_out_mask = np.ones((self.n_neurons, self._n_output), dtype=np.float32)

        if self._EI_ratio and not (isinstance(connection_pars['w_rec'], np.ndarray) and np.any(connection_pars[
                                                                                                   'w_rec'] < 0)):
            logger.info("Setting E/I connections [g={}]".format(ei_balance))
            # if E/I are split and connection matrix is explicitly provided, it needs to contain positive and
            # negative weights, otherwise, inhibitory weights are added
            id_list = self.gids
            id_list[0] = 1
            ei_list = np.sign(id_list)
            inh_indices = np.where(ei_list == -1)[0]
            ei_matrix = np.diag(ei_list).astype(np.float32)

            w_rec = self.connections.synaptic_weights['w_rec']
            w_rec[:, inh_indices] *= -ei_balance
            # self.w_rec = tf.cast(tf.constant(ei_matrix) @ tf.nn.relu(w_rec), dtype=tf.float32) # ensure excitatory
            # # neurons only have postive outgoing weights, and inhibitory neurons have negative outgoing weights
            w_rec_mask = np.ones((self.n_neurons, self.n_neurons), dtype=np.float32) - np.eye(self.n_neurons)
            w_out_mask[inh_indices, :] = 0 # don't readout from I neurons
            w_out = self.connections.synaptic_weights['w_out']
            # print(type(w_rec), type(w_rec_mask))

            self.connections.set_connections(connection_name='w_rec', weights=(w_rec*w_rec_mask).astype(np.float32))
            self.connections.set_connections(connection_name='w_out', weights=(w_out*w_out_mask).astype(np.float32))
        else:
            ei_matrix = np.eye(self.n_neurons).astype(np.float32)

        for k, v in self.connections.synaptic_weights.items():
            self.variables.update({k: tf.Variable(v, name=k)})

        self.w_rec = tf.cast(tf.constant(ei_matrix) @ tf.nn.relu(self.variables['w_rec']), dtype=tf.float32)
        self.connection_masks = {'w_in': w_in_mask, 'w_rec': w_rec_mask, 'w_out': w_out_mask}

    def reset_graph(self):
        """
        Reset TensorFlow before running!
        """
        if self._sess:
            self._sess.close()
        tf.reset_default_graph()

    def build_graph(self):
        """
        Initialize all weights, biases and states
        """
        logger.info("Preparing system variables")
        self.variables = {
            'hidden_state': tf.Variable(self.initialize_states(), 'x'),
            'active_state': tf.Variable(self.initialize_states(), 'r'),
            'bias_rec': tf.Variable(np.zeros((1, self.n_neurons), dtype=np.float32), name='b'),
            'bias_out': tf.Variable(np.zeros((1, self._n_output), dtype=np.float32), name='b_out')}

    def get_states(self, batch_state, input_times=None, dataset_label=None):
        """
        Store population states
        :param batch_state:
        :param input_times:
        :param dataset_label:
        :return:
        """
        if input_times is not None and isinstance(input_times[0], tuple):
            stim_onsets = [x[0] for x in input_times]
            stim_offsets = [x[1] for x in input_times]
            batch_time_axis = np.arange(stim_onsets[0], stim_offsets[-1], self._resolution)
        else:
            batch_time_axis = np.arange(0., batch_state.shape[1], 1.)
        sampled_times = batch_time_axis

        for ext_label, ext_params in self._state_extractors.items():
            state_matrix = batch_state
            if 'sampling_times' in ext_params.keys():
                # sample states at offset
                if ext_params['sampling_times'] == ['stim_offset']:
                    sampling_indices = [np.where(np.round(batch_time_axis, 1) == x-self._resolution)[0][0]
                                        for x in stim_offsets]
                    sampled_times = [x-self._resolution for x in stim_offsets]
                else:
                    sampling_indices = [np.where(np.round(batch_time_axis, 1) == x)[0][0]
                                        for x in ext_params['sampling_times']]
                    sampled_times = [x for x in ext_params['sampling_times']]
                state = state_matrix[:, sampling_indices]
            elif 'sampling_rate' in ext_params.keys():
                state = state_matrix[:, ::ext_params['sampling_rate']]

            self.dec_states.update({ext_label: StateMatrix(state, ext_label, state_var=ext_params['variable'],
                                                       population=self.name, sampled_times=sampled_times,
                                                       dataset_label=dataset_label, save=ext_params['save'])})
            if "standardize" in ext_params.keys():
                if ext_params['standardize']:
                    self.dec_states[ext_label].standardize()

    def optimize(self, target, weight_penalty=0., rate_penalty=0., clip_max_gradient=0.1):
        """
        Calculate the losses and optimize
        :param weight_penalty: [float]
        :param rate_penalty: [float]
        """
        #TODO - add biases, timeconstants to tunable parameters
        self.tf_data.update({
            # 'perf_loss': tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.outputs,
            #                                                                        labels=target))})#,axis=2))})
            'perf_loss': tf.reduce_mean(tf.squared_difference(self.outputs, target))})
        total_loss = self.tf_data['perf_loss']
        self.tf_data.update({'weight_loss': tf.reduce_mean(tf.nn.relu(self.w_rec)**2)})
        total_loss += self.tf_data['weight_loss'] * weight_penalty
        self.tf_data.update({'rate_loss': tf.reduce_mean(self.states**2)})
        total_loss += self.tf_data['rate_loss'] * rate_penalty

        self.tf_data.update({'total_loss': total_loss})
        optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        gradients = optimizer.compute_gradients(self.tf_data['total_loss'])

        # clip gradients
        capped_rvs = []
        for grad, var in gradients:
            if 'w_rec' in var.op.name:
                grad *= self.connection_masks['w_rec']
            elif 'w_in' in var.op.name:
                grad *= self.connection_masks['w_in']
            elif 'w_out' in var.op.name:
                grad *= self.connection_masks['w_out']
            capped_rvs.append((tf.clip_by_norm(grad, clip_max_gradient), var))

        self.tf_data.update({'optimizer': optimizer.apply_gradients(capped_rvs)})

    @staticmethod
    def rnn_cell(alpha, w_in, w_rec, b_rec, noise, rnn_input, h, transfer_function='tanh'):
        # TODO add transfer functions, compute hidden and active states (x/r)
        # no synaptic plasticity (otherwise just add equations here)
        h_pre = h

        # Update the hidden state.
        if transfer_function == 'relu':
            h = tf.nn.relu(h * (1-alpha) + alpha * (rnn_input @ tf.nn.relu(w_in) + h_pre @ w_rec + b_rec)
                           + tf.random_normal(h.shape, 0, noise, dtype=tf.float32))
        elif transfer_function == 'tanh':
            h = tf.nn.tanh(h * (1-alpha) + alpha * (rnn_input @ tf.nn.relu(w_in) + h_pre @ w_rec + b_rec)
                           + tf.random_normal(h.shape, 0, noise, dtype=tf.float32))
        return h

    @staticmethod
    def complete_rnn_cell(alphas, w_in, w_rec, b_rec, noise, rnn_input, states, transfer_function='tanh'):
        # TODO add transfer functions, compute hidden and active states (x/r)
        # no synaptic plasticity (otherwise just add equations here)
        x_pre = states[0]
        r_pre = states[1]

        x = x_pre * (1 - alphas[0]) + alphas[0] * (r_pre @ w_rec + rnn_input @ w_in + b_rec +
                                                   tf.random_normal(r_pre.shape, 0, noise,
                                                                    dtype=tf.float32))

        if transfer_function == 'relu':
            r = r_pre * (1-alphas[1]) + alphas[1] * tf.nn.relu(x)
        elif transfer_function == 'tanh':
            r = r_pre * (1-alphas[1]) + alphas[1] * tf.nn.tanh(x)
        elif transfer_function == 'sigmoid':
            r = r_pre * (1-alphas[1]) + alphas[1] * tf.nn.sigmoid(x)
        return x, r

    def prepare_for_batch_processing(self, T, batch_size):
        """
        Setup placeholders for the input and output tensors
        :param T: length of time axis
        :param batch_size: currently needs to be fixed
        :return:
        """
        u = tf.placeholder(tf.float32, [T, batch_size, self._n_input], name='input')
        z = tf.placeholder(tf.float32, [T, batch_size, self._n_output], name='target')
        return u, z

    def run_model(self, input_data): # process_batch
        """
        Main model loop
        :return:
        """
        self.states = []
        self.outputs = []
        x = self.variables['hidden_state']
        r = self.variables['active_state']

        alphas = (np.float32(self._resolution / self._taus[0]), np.float32(self._resolution / self._taus[1]))

        # Loop through the neural inputs to the RNN, indexed in time
        for rnn_input in input_data:
            # self.states.append(r)
            # h = self.rnn_cell(alpha=alphas, w_in=self.variables['w_in'], w_rec=self.w_rec,
            #                   b_rec=self.variables['bias_rec'], noise=self._noise, rnn_input=rnn_input, h=h,
            #                   transfer_function=self._trf)
            x, r = self.complete_rnn_cell(alphas=alphas, w_in=self.variables['w_in'], w_rec=self.w_rec,
                                          b_rec=self.variables['bias_rec'], noise=self._noise, rnn_input=rnn_input,
                                          states=(x, r), transfer_function=self._trf)
            self.states.append(r)
            self.outputs.append(r @ tf.nn.relu(self.variables['w_out']) + self.variables['bias_out'])

        self.states = tf.stack(self.states)
        self.outputs = tf.stack(self.outputs)

    def train(self, data_batch, n_epochs=1, weight_penalty=0., rate_penalty=0., clip_max_gradient=0.1,
              gpu_id=None, symbolic=True, verbose=True, save=False, label=''):

        u, z = self.prepare_for_batch_processing(T=data_batch['inputs'].shape[0], batch_size=data_batch['inputs'].shape[1])
        self._n_batches = data_batch['inputs'].shape[1]
        self.tf_data.update({'input': u, 'output': z})

        inpt = tf.unstack(u, axis=0)
        tget = z#tf.unstack(z, axis=0)

        saver = tf.train.Saver()
        states = []
        outputs = []
        dec_outputs = []
        dec_targets = []
        training_losses = []
        dec_loss = []
        decoder_outputs = data_batch['decoder_outputs']

        with tf.Session(config=tf.ConfigProto()) as self._sess:
            device = '/cpu:0' if gpu_id is None else '/gpu:0'
            logger.info("Running TF session on {}".format(device))
            with tf.device(device):
                self.run_model(inpt)
                self.optimize(tget, weight_penalty, rate_penalty, clip_max_gradient)

            self._sess.run(tf.global_variables_initializer())

            for epoch in range(n_epochs):
                evaluate = [self.tf_data['optimizer'], self.tf_data['total_loss'], self.tf_data['perf_loss'],
                            self.tf_data['rate_loss'], self.tf_data['weight_loss'], self.outputs, self.states]

                (_, loss, perf_loss, rate_loss, weight_loss, output, state) = self.process_batch(evaluate, data_batch)

                if self.decoders is not None:
                    for n_batch in range(self._n_batches):
                        batch_label = "train_batch={}_epoch={}".format(n_batch+1, epoch+1)
                        st = state[:, n_batch, :].T

                        self.get_states(st, input_times=data_batch['input_times'], dataset_label=batch_label)

                        if not self.decoders.initialized:
                            self.decoders.connect(self._extractor_labels, decoder_outputs[n_batch])

                        self.decoders.train(self.dec_states, batch_label, decoder_outputs[n_batch],
                                            stim_info=data_batch['input_times'])
                    self.decoders.validate(batch_label, symbolic)
                    if epoch == n_epochs - 1:
                        dec_loss = self.decoders.validation_accuracy
                        perf = self.decoder_accuracy(output_parsing='k-WTA', symbolic_task=symbolic)

                if verbose:
                    logger.info("Epoch {0} loss: {1}".format(epoch, loss))
                training_losses.append(loss)

                if epoch == 0 or epoch == n_epochs-1:
                    states.append(state)
                    outputs.append(output)
                    if self.decoders is not None:
                        outs, tgts = self.decoders.retrieve_outputs()
                        dec_outputs.append(outs)
                        dec_targets.append(tgts)

            trained_coeffs = self._sess.run(self.variables)

            if save:
                pth = utils.data_handling.paths['other'] + label + '/'
                if not os.path.exists(pth):
                    os.makedirs(pth)
                saver.save(self._sess, pth)
                if self.decoders is not None:
                    self.decoders.save_training_data()

            return {'losses': training_losses, 'states': states, 'outputs': outputs,
                    'trained_parameters': trained_coeffs, 'decoder_loss': dec_loss, 'decoder_accuracy': perf,
                    'decoder_outputs': dec_outputs, 'decoder_targets': dec_targets}

    def test(self, data_batch, output_parsing='k-WTA', symbolic=True, verbose=True, save=False, gpu_id=None, label=''):
        """
        Process a test set
        :param data_batch:
        :param output_parsing:
        :param symbolic:
        :return:
        """
        test_results = self.predict(data_batch, verbose, gpu_id, save=save, label=label)

        decoder_outputs = data_batch['decoder_outputs']
        perf = []
        if self.decoders is not None:
            n_batch = -1
            batch_label = "test_batch={}".format(n_batch)
            st = test_results['states'][:, n_batch, :].T
            self.get_states(st, input_times=data_batch['input_times'], dataset_label=batch_label)
            self.decoders.predict(self.dec_states, decoder_outputs[n_batch],
                                  stim_info=data_batch['input_times'])
            perf = self.decoder_accuracy(output_parsing=output_parsing, symbolic_task=symbolic)
            outs, tgts = self.decoders.retrieve_outputs()
            test_results.update({'decoder_outputs': outs, 'decoder_targets': tgts})
        test_results.update({'decoder_accuracy': perf})
        return test_results

    def predict(self, data_batch, verbose=True, gpu_id=None, save=False, label=''):
        """
        Runs the network without training (pass data batch and generate outputs)
        :return:
        """
        saver = tf.train.Saver(tf.trainable_variables()) # tf.train.Saver()
        pth = utils.data_handling.paths['other'] + label + '/'
        chkpt_ = tf.train.get_checkpoint_state(pth)

        with tf.Session(config=tf.ConfigProto()) as self._sess:
            device = '/cpu:0' if gpu_id is None else '/gpu:0'
            logger.info("Running TF session on {}".format(device))

            self._sess.run(tf.global_variables_initializer())
            logger.info("Restoring from checkpoint")
            saver.restore(self._sess, chkpt_.model_checkpoint_path)

            evaluate = [self.states, self.outputs]
            (hidden_state, predict) = self.process_batch(evaluate, data_batch)

            if save:
                pth = utils.data_handling.paths['other']
                saver.save(self._sess, pth)

        return {'states': hidden_state, 'outputs': predict}

    def process_batch(self, ev_options, data_batch):
        """
        :return:
        """
        session_outputs = self._sess.run(ev_options, feed_dict={self.tf_data['input']: data_batch['inputs'],
                                                                self.tf_data['output']: data_batch['targets']})
        return session_outputs

    def decoder_accuracy(self, output_parsing='k-WTA', symbolic_task=True, store=True):
        """
        Evaluate the accuracy of the decoders on the test set
        """
        return self.decoders.evaluate(process_output_method=output_parsing, symbolic=symbolic_task,
                                      flush=not store, mse_only=not symbolic_task)
