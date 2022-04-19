import numpy as np
from tqdm import tqdm

from fna.decoders import Decoder, StateMatrix

from fna.tools.network_architect import Network
from fna.tools.network_architect.connectivity import TFConnector
from fna.tools.parameters import ParameterSet
from fna.tools import utils

logger = utils.logger.get_logger(__name__)


class ReservoirRecurrentNetwork(Network):
    """
    Wrapper for rate network models implemented in pure Python (special cases). Networks are not trained,
    only the readouts
    """

    def report(self):
        """
        Print a description of the system
        """
        logger.info("========================================================")
        logger.info(" {0!s} architecture ({1!s}-simulated):".format(self.name, self.simulator))
        logger.info("--------------------------------------------------------")
        logger.info("- Size: {0!s}".format(self.n_neurons))
        logger.info("- Neuron models: {0!s}".format(self._trf))

    def __init__(self, network_parameters, connection_parameters, input_dim, output_dim, dt=0.1, label=None,
                 extractors=None, decoders=None):
        """

        """
        self.name = label
        self.simulator = "Python"
        logger.info("Initializing {0!s} architecture ({1!s}-simulated)".format(self.name, self.simulator))
        self._parse_parameters(network_parameters, parameter_type='network')
        self._n_input = input_dim
        self._n_output = output_dim
        self.gids = np.arange(self.n_neurons).astype(int)
        self._resolution = dt
        self.variables = {}

        if self._EI_ratio:
            nE = self.n_neurons * self._EI_ratio
            ei_list = np.ones(self.n_neurons, dtype=int)
            ei_list[int(nE):] = -1
            self.gids *= ei_list # negative gids will correspond to inh neurons

        self.states = {}
        self.outputs = None
        self.connections = None
        self.w_rec = None
        self.w_in = None
        self.decoders = None

        self.report()
        if isinstance(connection_parameters['w_rec'], dict):
            ei_balance = connection_parameters['w_rec']['ei_balance']
        else:
            ei_balance = 0.
        self.connect(connection_parameters, ei_balance=ei_balance)

        self.build_graph()

        if extractors is not None:
            self.set_extractors(extractors)
        if decoders is not None:
            self.create_decoders(decoders)

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
            if isinstance(parameters.tau_x, np.ndarray):
                self._taus = (parameters.tau_x, parameters.tau_r)
            elif isinstance(parameters.tau_x, float):
                self._taus = (np.ones(self.n_neurons)*parameters.tau_x, np.ones(self.n_neurons)*parameters.tau_r)
            elif isinstance(parameters.tau_x, tuple):
                self._taus = (eval(parameters.tau_x[0])(**parameters.tau_x[1]),
                              eval(parameters.tau_r[0])(**parameters.tau_r[1]))
            self._noise = parameters.noise
            self._initialize_states = parameters.initial_state
            # self._learning_rate = parameters.learning_rate
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
                # elif isinstance(v, np.ndarray):
                #     assert np.shape(v) == (self._n_input, self.n_neurons), "Incorrect shape for weight matrix {}".format(k)
                # else:
                #     raise NotImplementedError("Weight matrices as {} cannot be used".format(str(type(v))))
            return parameters

    def initialize_states(self, var_label):
        """
        Set the initial states
        """
        logger.info("Initializing {} state variable".format(var_label))
        if self._initialize_states is None:
            return 0.1 * np.ones((1, self.n_neurons), dtype=np.float32)
        else:
            assert isinstance(self._initialize_states, tuple), "Initial states must be a tuple or None (for default " \
                                                               "initialization)"
            return self._initialize_states[0](size=(1, self.n_neurons), **self._initialize_states[1]).astype(np.float32)

    def build_graph(self):
        """
        Initialize all weights, biases and states
        """
        logger.info("Preparing system variables:")
        self.variables = {
            'hidden_state': self.initialize_states(var_label='hidden'),
            'active_state': self.initialize_states(var_label='active')}

    def connect(self, connection_pars, ei_balance=1.):
        """
        Setup connection weights
        :param connection_pars: dict - must contain "w_in", "w_rec" and "w_out"
        :param ei_balance: if connections are split into E/I, how much stronger should I be?
        """
        connection_pars = self._parse_parameters(connection_pars, parameter_type="connections")
        # print(connection_pars)
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

        self.w_rec = ei_matrix @ self.connections.synaptic_weights['w_rec']
        self.w_in = self.connections.synaptic_weights['w_in']

    def create_decoders(self, readout_params):
        """
        Create decoders: list of Readouts
        """
        self.decoders = Decoder(readout_params)

    def set_extractors(self, extractor_params):
        """
        Determine which state variables to store and how
        :param extractor_params:
        :return:
        """
        self._state_extractors = extractor_params
        self._extractor_labels = list(extractor_params.keys())

    def get_states(self, batch_state, input_times=None, dataset_label=None):
        """
        Gather and store population states
        :param batch_state:
        :param input_times:
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
            state_matrix = batch_state[ext_params['variable']]
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

            self.states.update({ext_label: StateMatrix(state, ext_label, state_var=ext_params['variable'],
                                                       population=self.name, sampled_times=sampled_times,
                                                       dataset_label=dataset_label, save=ext_params['save'])})
            if "standardize" in ext_params.keys():
                if ext_params['standardize']:
                    self.states[ext_label].standardize()

    def process_batch(self, batch_label, input_batch):
        """
        Run simulation
        :param input_batch:
        :return:
        """
        logger.info("Processing batch {0}".format(batch_label))
        time_axis = input_batch.shape[1]

        x = self.variables['hidden_state']
        r = self.variables['active_state']
        state = {}
        state['active'] = np.empty((self.n_neurons, time_axis))
        state['hidden'] = np.empty((self.n_neurons, time_axis))

        alphas = (np.float32(self._resolution / self._taus[0]), np.float32(self._resolution / self._taus[1]))

        # update state
        for t in tqdm(range(time_axis), desc="Simulating: "):
            state['active'][:, t] = r
            state['hidden'][:, t] = x

            noise = np.random.normal(size=r.shape) * self._noise
            x = x * (1 - alphas[0]) + alphas[0] * (r @ self.w_rec + input_batch[:, t] @ self.w_in + noise)
            r = r * (1 - alphas[1]) + alphas[1] * self.update_state(x, self._trf)

        self.variables['hidden_state'] = x
        self.variables['active_state'] = r

        return state

    @staticmethod
    def update_state(x, transfer_fcn='tanh', rectify=False, n=None, th=None):
        """
        Compute unit rate from saturating nonlinearity
        :param x: system state
        :param transfer_fcn: transfer function, string
        :param n: exponent, if transfer_fcn is supralinear
        :param th: threshold if transfer_fcn is a step function
        :return:
        """
        if rectify:
            x = np.abs(x)

        if transfer_fcn == 'tanh':
            r = np.tanh(x)
        elif transfer_fcn == 'sigmoid':
            r = 1. / (1. + np.exp(-x))
        elif transfer_fcn == 'linear':  # rectified linear
            r = np.maximum(x, np.zeros_like(x))
        elif transfer_fcn == 'supralinear':
            assert n is not None and rectify, "Rectified supralinear transfer function requires rectify=True and the " \
                                              "order parameter n > 1"
            r = np.power(x, n)
        elif transfer_fcn == 'step':
            r = ((x - th) > 0.).astype(float)
        else:
            raise NotImplementedError("Transfer function {0} not implemented".format(transfer_fcn))
        return r

    def train(self, data_batch, n_epochs, symbolic=True, vocabulary=None, save=False, verbose=True):
        """
        Process training batch and train decoders
        """
        states = []
        outputs = []
        losses = []
        dec_loss = []
        dec_outputs = []
        dec_targets = []
        perf = None

        x_train = data_batch['inputs']
        input_times = data_batch['input_times']
        n_batches = x_train.T.shape[1]

        y_train = data_batch['decoder_outputs']

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                batch_label = 'train_batch={}_epoch={}'.format(batch+1, epoch+1)

                batch_input = x_train[:, batch, :].T
                batch_target = y_train[batch]

                state = self.fit(batch_label, data_batch=batch_input, input_times=input_times,
                                 target_outputs=batch_target, vocabulary=vocabulary)

            self.decoders.validate(batch_label, symbolic, vocabulary)
            if epoch == n_epochs - 1:
                dec_loss = self.decoders.validation_accuracy
                perf = self.decoder_accuracy(output_parsing="k-WTA" if symbolic else None, symbolic_task=symbolic,
                                             vocabulary=vocabulary)

            if epoch == 0 or epoch == n_epochs-1:
                states.append(state['active'])
                outputs.append(self.decoders.readouts[0].output)
                outs, tgts = self.decoders.retrieve_outputs()
                dec_outputs.append(outs)
                dec_targets.append(tgts)

        if save:
            self.decoders.save_training_data()

        return {'losses': losses, 'states': states, 'outputs': outputs, 'decoder_loss': dec_loss,
                'decoder_accuracy': perf, 'decoder_outputs': dec_outputs, 'decoder_targets': dec_targets}

    def test(self, data_batch, output_parsing='k-WTA', symbolic=True, vocabulary=None, save=False, verbose=True):
        """
        Process test set
        :param data_batch:
        :param output_parsing:
        :param symbolic:
        :param save_states:
        :return:
        """
        x_train = data_batch['inputs']
        input_times = data_batch['input_times']
        n_batches = x_train.T.shape[1]

        y_train = data_batch['decoder_outputs']

        for batch in range(n_batches):
            batch_label = 'test_batch={}'.format(batch+1)
            batch_input = x_train[:, batch, :].T
            batch_target = y_train[batch]

            state = self.predict(batch_label, data_batch=batch_input, input_times=input_times,
                                 target_outputs=batch_target, vocabulary=vocabulary)
        perf = self.decoder_accuracy(output_parsing=output_parsing, symbolic_task=symbolic, vocabulary=vocabulary)
        outs, tgts = self.decoders.retrieve_outputs()

        return {'states': state['active'], 'decoder_accuracy': perf, 'outputs':
            self.decoders.readouts[0].output, 'decoder_outputs': outs, 'decoder_targets': tgts}

    def fit(self, batch_label, data_batch, target_outputs, input_times=None, vocabulary=None):
        """

        """
        states = self.process_batch(batch_label=batch_label, input_batch=data_batch)
        self.get_states(states, input_times, dataset_label=batch_label)

        if not self.decoders.initialized:
            self.decoders.connect(self._extractor_labels, target_outputs)

        self.decoders.train(self.states, batch_label, target_outputs, stim_info=input_times, vocabulary=vocabulary)

        return states

    def predict(self, batch_label, data_batch, target_outputs, input_times=None, vocabulary=None):
        """

        """
        states = self.process_batch(batch_label=batch_label, input_batch=data_batch)
        self.get_states(states, input_times, dataset_label=batch_label)

        self.decoders.predict(self.states, target_outputs, stim_info=input_times, vocabulary=vocabulary)

        return states

    def decoder_accuracy(self, output_parsing='k-WTA', symbolic_task=True, store=True, vocabulary=None):
        """
        Evaluate the accuracy of the decoders on the test set
        """
        return self.decoders.evaluate(process_output_method=output_parsing, symbolic=symbolic_task,
                                      flush=not store, mse_only=not symbolic_task, vocabulary=vocabulary)

