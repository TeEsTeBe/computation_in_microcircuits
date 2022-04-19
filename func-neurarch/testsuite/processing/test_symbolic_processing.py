import numpy as np

# ###################################################################
# MNIST digit classification in example ANNs, cRNNs and SNNs
from fna.tasks.symbolic.sequences import SymbolicSequencer
# from fna.tasks.preprocessing import ImageFrontend
from fna.tasks.symbolic.embeddings import VectorEmbeddings
from fna.networks.rnn.helper import prepare_symbolic_batch

# system parameters
resolution = 1.

# task parameters
vocabulary_size = 2
n_epochs = 10
n_batches = 2
batch_size = 10
continuous = True

# discrete sequencers
sequencer = SymbolicSequencer(label='random sequence', set_size=vocabulary_size)
# image_mnist = ImageFrontend(path='../data/mnist/', label='mnist', vocabulary=sequencer.tokens)
emb = VectorEmbeddings(vocabulary=sequencer.tokens).binary_codeword(dim=10, density=0.1)

# continuous sequencers
signal_pars = {
    'duration': 20., # [ms]
    'amplitude': 1., # [max. rate]
    'kernel': ('alpha', {'tau': 25.}),# (kernel, {kernel_pars})
    'dt': resolution # [ms]
}
# image_mnist.unfold(to_signal=True, **signal_pars)
sig = emb.unfold(to_signal=True, **signal_pars)


# TODO integrate tensorflow
def tst_ANN():
    # ##################################################################################################################
    # 1) VanillaRNN
    # ##################################################################################################################
    from fna.networks.ann import ArtificialNeuralNetwork

    net_params = {
        'N': 500,
        'cell_type': 'LSTM',
        'learning_rate': 1e-2
    }
    rnn = ArtificialNeuralNetwork(label='LSTM', network_parameters=net_params,
                                  input_dim=emb.embedding_dimensions,
                                  output_dim=len(sequencer.tokens))

    train_batch = prepare_symbolic_batch(simulator=rnn.simulator, n_batches=n_batches, batch_size=batch_size,
                                         sequencer=sequencer, discrete_embedding=emb, as_tensor=False)
    train_results = rnn.train(train_batch, n_epochs=n_epochs, verbose=True, save=False)

    # test
    test_batch = prepare_symbolic_batch(simulator=rnn.simulator, n_batches=n_batches, batch_size=batch_size,
                                        sequencer=sequencer, discrete_embedding=emb, as_tensor=False)
    test_results, decoder_accuracy = rnn.test(test_batch, output_parsing='k-WTA', symbolic=True, verbose=True, save=False)

    # loss = train_results['losses']
    # states = train_results['states']
    # outputs = train_results['outputs']
    #
    # batch_time_axis = np.arange(0., batch_size, 1)
    # input_signal = train_batch['inputs'][-1].T
    # state_matrix = states[-1][-1, :, :].T
    # output = outputs[-1][-state_matrix.shape[1]:, :].T
    # target = train_batch['targets'][-1].T


def test_rRNN():
    # ##################################################################################################################
    # 2) Continuous rate RNN
    # ##################################################################################################################
    from fna.networks.rnn import ReservoirRecurrentNetwork

    N = 100
    taus = np.ones(N) * 100.
    noise = np.sqrt(2*(resolution/taus)) * 0.5
    batch_time = (signal_pars['duration'] * batch_size) / resolution

    net_params = {
        'N': N,
        'transfer_fcn': 'sigmoid',
        'EI_ratio': 0,
        'tau_x': taus,
        'tau_r': np.ones_like(taus)*resolution,
        'noise': noise.astype(np.float32),
        'initial_state': (np.random.uniform, {'low': 0., 'high': 1.}),
        'learning_rate': 1e-2
    }
    connection_params = {
        'w_in': {
            'density': 1.,
            'distribution': (np.random.uniform, {'low': -1., 'high': 1.})
        },
        'w_rec': {
            'ei_balance': 4.,
            'density': .1,
            'distribution': (np.random.gamma, {'shape': 0.2, 'scale': 1.})
        },
        'w_out': {
            'density': 1.,
            'distribution': (np.random.gamma, {'shape': 0.1, 'scale': 1.})
        }
    }

    extractor_parameters = {
        'r_@offset': {
            'population': None,
            'variable': 'active',
            'sampling_times': ['stim_offset'],
            'save': True}}
    decoding_parameters = {
        'SGD-Regression': {
            'algorithm': "sgd-reg",
            'extractor': 'r_@offset',
            'save': True
        },}

    # image_mnist.embedding_dimensions = image_mnist.dynamic_embedding.embedding_dimensions

    rnn = ReservoirRecurrentNetwork(label='rRNN-random-amorphous', network_parameters=net_params,
                                    connection_parameters=connection_params, input_dim=emb.embedding_dimensions,
                                    extractors=extractor_parameters, decoders=decoding_parameters,
                                    output_dim=vocabulary_size, dt=resolution)

    # Train set
    train_batch = prepare_symbolic_batch(simulator=rnn.simulator, n_batches=n_batches, batch_size=batch_size,
                                         sequencer=sequencer, continuous_embedding=sig,
                                         batch_time=int(batch_time), as_tensor=True, signal_pars=signal_pars,
                                         decoder_output_pars=None)
    train_results = rnn.train(data_batch=train_batch, n_epochs=n_epochs, symbolic=True, save=False,
                              vocabulary=sequencer.tokens)

    # Test set
    test_batch = prepare_symbolic_batch(simulator=rnn.simulator,
                                        n_batches=1, batch_size=batch_size,
                                        sequencer=sequencer, continuous_embedding=sig,
                                        batch_time=int(batch_time), as_tensor=True, signal_pars=signal_pars,
                                        decoder_output_pars=None)
    test_results = rnn.test(data_batch=test_batch, output_parsing='k-WTA', symbolic=True, save=False,
                            vocabulary=sequencer.tokens)

    # prepare variables for analysis
    # loss = [train_results['losses'][epoch]['Batch={}_epoch={}'.format(n_batches, epoch+1)]['{}-classification'.format(
    #     rnn.decoders.readouts[0].label)]['raw-MSE'] for epoch in range(n_epochs)]
    # states = train_results['states']
    # outputs = train_results['outputs']
    #
    # batch_time_axis = np.arange(0., signal_pars['duration'] * batch_size, resolution)
    # input_signal = train_batch['inputs'][:, -1, :].T
    # state_matrix = states[-1]
    # output = outputs[-1][0]
    # target = train_batch['targets'][:, -1, :].T


def test_SNN():
    # ##################################################################################################################
    # 3) SNN
    # ##################################################################################################################
    import nest
    from examples import example_defaults
    from fna.encoders import NESTEncoder, InputMapper

    vocabulary_size = 2
    n_epochs = 1
    n_batches = 2
    batch_size = 10
    continuous = True

    batch_time = (signal_pars['duration'] * batch_size) / resolution

    example_defaults.reset_kernel(resolution=0.1)
    rnn, rnn_recurrent_connections = example_defaults.default_network(N=100, record_spikes=True)

    encoder = NESTEncoder('inhomogeneous_poisson_generator', label='poisson-input',
                          dim=emb.embedding_dimensions)
    # input synapses
    input_synapses = {
        'connect_populations': [('E', encoder.name), ('I', encoder.name), ],
        'weight_matrix': [None, None],
        'conn_specs': [{'rule': 'pairwise_bernoulli', 'p':0.1}, {'rule': 'pairwise_bernoulli', 'p':0.1}],
        'syn_specs': [{'model': 'static_synapse', 'delay': 0.1, 'weight': 1.},
                      {'model': 'static_synapse', 'delay': 0.1, 'weight': 1.}]
    }
    in_to_rnn_connections = InputMapper(source=encoder, target=rnn, parameters=input_synapses)

    # decoder
    extractor_parameters = {
        'E_Vm_@offset': {
            'population': 'E',
            'variable': 'V_m',
            'sampling_times': ['stim_offset'],
            'save': False}}
    decoding_parameters = {
        'SGD-Regression': {
            'algorithm': "sgd-reg",
            'extractor': 'E_Vm_@offset',
            'save': False
        },}

    # create and connect extractors and decoders
    rnn.connect_state_extractors(extractor_parameters, encoder=encoder, input_mapper=in_to_rnn_connections,
                                 stim_duration=signal_pars['duration'], stim_isi=None, stim_onset=rnn.next_onset_time(),
                                 to_memory=True)
    rng = np.random.default_rng(1234)
    rnn.create_decoder(decoding_parameters, rng=rng)
    total_delay = in_to_rnn_connections.total_delay + rnn.state_extractor.max_extractor_delay

    # train
    train_batch = prepare_symbolic_batch(simulator=rnn.simulator, n_batches=n_batches, batch_size=batch_size,
                                             sequencer=sequencer, continuous_embedding=sig,
                                             batch_time=int(batch_time), as_tensor=True, signal_pars=signal_pars,
                                             decoder_output_pars=None)
    train_results = rnn.train(train_batch, n_epochs, sig, encoder, rnn.next_onset_time(),
                              total_delay=total_delay, symbolic=True, continuous=continuous, verbose=True, save=False)

    # test
    test_batch = prepare_symbolic_batch(simulator=rnn.simulator, n_batches=1, batch_size=batch_size,
                                             sequencer=sequencer, continuous_embedding=sig,
                                             batch_time=int(batch_time), as_tensor=True, signal_pars=signal_pars,
                                             decoder_output_pars=None)
    test_results = rnn.test(test_batch, sig, encoder, rnn.next_onset_time(), total_delay=total_delay,
                            symbolic=True, continuous=continuous, output_parsing="k-WTA")

    # prepare variables for analysis
    # loss = [train_results['losses'][epoch]['Batch={}_epoch={}'.format(n_batches, epoch+1)]['{}-classification'.format(
    #     rnn.decoders.readouts[0].label)]['raw-MSE'] for epoch in range(n_epochs)]
    # states = train_results['states']
    # outputs = train_results['outputs']
    #
    # batch_time_axis = np.arange(0., signal_pars['duration'] * batch_size, resolution)
    # input_signal, _ = image_mnist.draw_stimulus_sequence(train_batch['inputs'][-1], continuous=True, unfold=True,
    #                                                      onset_time=0., verbose=False)
    # input_signal = input_signal.as_array()
    # state_matrix = states[-1][0].matrix
    # output = outputs[-1][0]
    #
    # target_seq = train_batch['decoder_outputs'][-1][0]['output']
    # out_signal = VectorEmbeddings(vocabulary=np.unique(target_seq)).one_hot()
    # target = out_signal.draw_stimulus_sequence(target_seq, as_array=True, verbose=False)

# test_rRNN()
# test_SNN()
