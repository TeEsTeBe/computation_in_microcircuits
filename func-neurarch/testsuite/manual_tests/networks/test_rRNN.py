# import tensorflow as tf
# import nest
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import types

from tools.parameters import ParameterSet

# ###################################################################
# MNIST digit classification in example ANNs, cRNNs and SNNs
from tasks.symbolic.sequences import SymbolicSequencer
from tasks.preprocessing import ImageFrontend
from tasks.symbolic.embeddings import VectorEmbeddings

# system parameters
resolution = 1.

# task parameters
vocabulary_size = 5
n_epochs = 5
n_batches = 10
batch_size = 100

# discrete sequencers
sequencer = SymbolicSequencer(label='random sequence', set_size=vocabulary_size)
image_mnist = ImageFrontend(path='/home/neuro/Desktop/dev/data/mnist/', label='mnist', vocabulary=sequencer.tokens)

# continuous sequencers
signal_pars = {
    'duration': 10., # [ms]
    'amplitude': 1., # [max. rate]
    'kernel': ('alpha', {'tau': 25.}),# (kernel, {kernel_pars})
    'dt': resolution # [ms]
}
image_mnist.unfold(to_signal=True, **signal_pars)


# ######################################################################################################################
def prepare_tf_batch(n_batches, batch_size, sequencer, discrete_embedding=None, continuous_embedding=None, batch_time=None,
                     as_tensor=False, signal_pars=None):
    """
    Generates data batches for tensorflow simulations
    :param n_batches: Number of unique batches
    :param batch_size: Size of batch (discrete steps or time)
    """
    # TODO - make more flexible (invert dimensions)...
    if not as_tensor:
        inputs = []
        targets = []
    else:
        inputs = np.empty(shape=(batch_time, int(n_batches), int(continuous_embedding.embedding_dimensions)),
                          dtype=np.float32)
        targets = np.empty(shape=(batch_time, int(n_batches), int(len(sequencer.tokens))), dtype=np.float32)

    for batch in tqdm(range(n_batches), desc="Generating batches"):
        # inputs
        batch_seq = sequencer.generate_random_sequence(T=batch_size, verbose=False)
        print(batch_seq)
        if discrete_embedding is not None and not as_tensor:
            input_batch = discrete_embedding.draw_stimulus_sequence(batch_seq, as_array=True, verbose=False)
            inputs.append(input_batch.T)

        elif continuous_embedding is not None and as_tensor:
            input_batch = continuous_embedding.draw_stimulus_sequence(batch_seq, continuous=True, unfold=True,
                                                                      onset_time=0., verbose=False)[0]
            if isinstance(input_batch, types.GeneratorType):
                for t, inp in enumerate(input_batch):
                    inputs[int(inp[1][0]/inp[0].dt):int(inp[1][1]/inp[0].dt), batch, :] = inp[0].as_array().T
            else:
                inputs[:, batch, :] = input_batch.as_array().astype(np.float32).T

        # targets
        target_seq = sequencer.generate_default_outputs(batch_seq, max_memory=0, max_chunk=0, max_prediction=0)[0][
            'output']  # classification !!

        if discrete_embedding is not None:
            target_batch = VectorEmbeddings(vocabulary=sequencer.tokens).one_hot().draw_stimulus_sequence(target_seq,
                                                                                                          as_array=True,
                                                                                                          verbose=False)
            targets.append(target_batch.T)
        elif continuous_embedding is not None:
            output_signal_pars = (lambda a, b: a.update(b) or a)(signal_pars, {'kernel': ('box', {}), 'amplitude': 1.})
            out_signal = VectorEmbeddings(vocabulary=np.unique(target_seq)).one_hot().unfold(to_signal=True,
                                                                                             verbose=False,
                                                                                             **output_signal_pars)
            target_batch = out_signal.draw_stimulus_sequence(target_seq, continuous=True, verbose=False)
            targets[:, batch, :] = target_batch[0].as_array().astype(np.float32).T

    return {'inputs': inputs, 'targets': targets}


def prepare_nest_batch(n_batches, batch_size, sequencer):
    """

    """
    batch_sequence = [sequencer.generate_random_sequence(T=batch_size, verbose=False) for _ in range(n_batches)]
    batch_targets = [sequencer.generate_default_outputs(batch_seq, max_memory=0, max_prediction=0,
                                                        max_chunk=0, chunk_memory=False, chunk_prediction=False) for batch_seq in batch_sequence]

    return batch_sequence, batch_targets


# ######################################################################################################################
# 2) Continuous rate RNN
# ######################################################################################################################
from networks.rnn import ReservoirRecurrentNetwork

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

image_mnist.embedding_dimensions = image_mnist.dynamic_embedding.embedding_dimensions

rnn = ReservoirRecurrentNetwork(label='ContinuousRNN',
                                network_parameters=net_params,
                                connection_parameters=connection_params,
                                input_dim=image_mnist.embedding_dimensions,
                                output_dim=vocabulary_size, dt=resolution)

# Transient set - to discard
# transient_batch_seq, _ = prepare_nest_batch(n_batches=1, batch_size=batch_size, sequencer=sequencer)
# transient_batch = image_mnist.draw_stimulus_sequence(transient_batch_seq[0], as_array=False, unfold=True, onset_time=0.,
#                                                      continuous=True, intervals=None, verbose=True)
# _ = rnn.process_batch(batch_label='transient', input_batch=transient_batch)

# set decoders
decoding_parameters = [ParameterSet({'algorithm': 'pinv', 'task': 'classification', 'extractor': None, 'save': True})]
rnn.set_decoders(decoding_parameters)


# Train set
train_batch = prepare_tf_batch(n_batches=n_batches, batch_size=batch_size, sequencer=sequencer,
                               continuous_embedding=image_mnist, batch_time=int(batch_time), as_tensor=True,
                               signal_pars=signal_pars)
train_results = rnn.train(x_train=train_batch['inputs'], y_train=train_batch['targets'],
                          n_batches=n_batches, n_epochs=n_epochs)

loss = train_results['losses']
states = train_results['states']
outputs = train_results['outputs']

# ######################################################################################################################
fig, ax = plt.subplots()
ax.plot(loss)

fig1 = plt.figure()
fig1.suptitle("States")
ax1 = fig1.add_subplot(211)
ax2 = fig1.add_subplot(212)

ax1.plot(states[0][0, :])
ax1.plot(states[0][10, :])
ax1.plot(states[0][30, :])

ax2.plot(states[-1][0, :])
ax2.plot(states[-1][10, :])
ax2.plot(states[-1][30, :])

fig2 = plt.figure()
fig2.suptitle("Outputs")
ax21 = fig2.add_subplot(211)
ax22 = fig2.add_subplot(212)
ax21.plot(outputs[0][0][:, 0], 'k', label='output')
ax21.plot(train_batch['targets'][:, 0, 0], 'r', label='target')
ax22.plot(outputs[-1][0][:, 0], 'k', label='output')
ax22.plot(train_batch['targets'][:, -1, 0], 'r', label='target')
plt.legend()
plt.show()

# ################################################################################
from decoders.readouts import Readout
from tools.parameters import ParameterSet
readout_pars = ParameterSet({'algorithm': 'pinv',
                             'task': 'classification',
                             'extractor': None})
readout1 = Readout(label='{}-readout'.format(rnn.name), readout_parameters=readout_pars)
state_matrix = states[-1]
target = train_batch['targets'][:, -1, :].T
readout1.train(batch_label='last-train-batch', state_matrix_train=state_matrix, target_train=target)
readout1.predict(state_matrix_test=state_matrix, target_test=target)
perf_1 = readout1.evaluate(process_output_method='k-WTA', symbolic=True)
print(perf_1)

readout1.output = outputs[-1][0]
perf_2 = readout1.evaluate(process_output_method='k-WTA', symbolic=True)
print(perf_2)

# single-sample identity
single_states = state_matrix[:, ::int(signal_pars['duration']/resolution)]
single_targets = target[:, ::int(signal_pars['duration']/resolution)]
readout_pars = ParameterSet({'algorithm': 'pinv',
                             'task': 'classification',
                             'extractor': None})
readout2 = Readout(label='{}-singe-sample-readout'.format(rnn.name), readout_parameters=readout_pars)
readout2.train(batch_label='single_sample-batch', state_matrix_train=single_states, target_train=single_targets)
readout2.predict(state_matrix_test=single_states, target_test=single_targets)
perf_3 = readout2.evaluate(process_output_method='k-WTA', symbolic=True)
print(perf_3)
