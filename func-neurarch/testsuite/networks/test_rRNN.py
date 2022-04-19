import numpy as np

from fna.tools.parameters import ParameterSet
from fna.networks.rnn import ReservoirRecurrentNetwork
from fna.networks.rnn.helper import prepare_symbolic_batch
from fna.decoders.readouts import Readout

# ###################################################################
# MNIST digit classification in example ANNs, cRNNs and SNNs
from fna.tasks.symbolic.sequences import SymbolicSequencer
# from fna.tasks.preprocessing import ImageFrontend
from fna.tasks.symbolic.embeddings import VectorEmbeddings

# system parameters
resolution = 1.

# task parameters
vocabulary_size = 2
n_epochs = 3
n_batches = 5
batch_size = 10

# discrete sequencers
sequencer = SymbolicSequencer(label='random sequence', set_size=vocabulary_size)
# image_mnist = ImageFrontend(path='../data/mnist/', label='mnist', vocabulary=sequencer.tokens)
emb = VectorEmbeddings(vocabulary=sequencer.tokens).binary_codeword(dim=10, density=0.1)

# continuous sequencers
signal_pars = {
    'duration': 10., # [ms]
    'amplitude': 1., # [max. rate]
    'kernel': ('alpha', {'tau': 25.}),# (kernel, {kernel_pars})
    'dt': resolution # [ms]
}
# image_mnist.unfold(to_signal=True, **signal_pars)
sig = emb.unfold(to_signal=True, **signal_pars)


# TODO rename when ready
def test():
    N = 100
    taus = np.ones(N) * 100.
    noise = np.sqrt(2*(resolution/taus)) * 0.5
    batch_time = (signal_pars['duration'] * batch_size) / resolution

    net_params = {
        'N': N,
        'transfer_fcn': 'sigmoid', # tanh, relu, sigmoid
        'EI_ratio': 0, # set ==0 if no division of E/I is necessary
        'tau_x': taus,
        'tau_r': np.ones_like(taus)*resolution,
        'noise': noise.astype(np.float32),
        'initial_state': (np.random.uniform, {'low': 0., 'high': 1.}),
        'learning_rate': 1e-2
    }
    connection_params = {
        'w_in': {
            'density': 1.,
            'distribution': (np.random.uniform, {'low': -1., 'high': 1.}) #(np.random.gamma, {'shape': 0.1, 'scale': 1.})
        }, # np.array(nU,N) or (distribution, {parameters})
        'w_rec': {
            'ei_balance': 4., # wI = ei_balance * wE
            'density': .1,
            'distribution': (np.random.gamma, {'shape': 0.2, 'scale': 1.})
        },
        'w_out': {
            'density': 1.,
            'distribution': (np.random.gamma, {'shape': 0.1, 'scale': 1.})
        }
    }

    # image_mnist.embedding_dimensions = image_mnist.dynamic_embedding.embedding_dimensions

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

    loss = train_results['losses']
    states = train_results['states']
    outputs = train_results['outputs']

    # # ######################################################################################################################
    # fig, ax = plt.subplots()
    # ax.plot(loss)
    #
    # fig1 = plt.figure()
    # fig1.suptitle("States")
    # ax1 = fig1.add_subplot(211)
    # ax2 = fig1.add_subplot(212)
    #
    # ax1.plot(states[0][0, :])
    # ax1.plot(states[0][10, :])
    # ax1.plot(states[0][30, :])
    #
    # ax2.plot(states[-1][0, :])
    # ax2.plot(states[-1][10, :])
    # ax2.plot(states[-1][30, :])
    #
    # fig2 = plt.figure()
    # fig2.suptitle("Outputs")
    # ax21 = fig2.add_subplot(211)
    # ax22 = fig2.add_subplot(212)
    # ax21.plot(outputs[0][0][:, 0], 'k', label='output')
    # ax21.plot(train_batch['targets'][:, 0, 0], 'r', label='target')
    # ax22.plot(outputs[-1][0][:, 0], 'k', label='output')
    # ax22.plot(train_batch['targets'][:, -1, 0], 'r', label='target')
    # plt.legend()
    # plt.show()

    # ################################################################################
    readout_pars = ParameterSet({'algorithm': 'pinv',
                                 'task': 'classification',
                                 'extractor': None})
    readout1 = Readout(label='{}-readout'.format(rnn.name), readout_parameters=readout_pars)
    state_matrix = states[-1]
    target = train_batch['targets'][:, -1, :].T
    readout1.train(batch_label='last-train-batch', state_matrix_train=state_matrix, target_train=target,
                   vocabulary=sequencer.tokens)
    readout1.predict(state_matrix_test=state_matrix, target_test=target, vocabulary=sequencer.tokens)
    perf_1 = readout1.evaluate(process_output_method='k-WTA', symbolic=True, vocabulary=sequencer.tokens)
    print(perf_1)
