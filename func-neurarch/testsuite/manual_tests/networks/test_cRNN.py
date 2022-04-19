import tensorflow as tf
from fna.tasks.symbolic.embeddings import VectorEmbeddings
from fna.tasks.symbolic.sequences import SymbolicSequencer
from fna.tasks.symbolic import NonAdjacentDependencies

import numpy as np
from tqdm import tqdm

# system properties
resolution = 1.

# ####################################################################
# input properties
# ###################################################################
n_epochs = 1000
n_batches = 100
batch_size = 100  # T [number of sequence steps]

sequencer = SymbolicSequencer(label='random', set_size=2)

# vector / scalar embeddings
emb = VectorEmbeddings(vocabulary=sequencer.tokens).binary_codeword(dim=10, density=0.1)

signal_pars = {
    'duration': 2., # single values or rvs
    'amplitude': 10., # single value, list of dim, or rvs
    'kernel': ('box', {}),# ('box', {}), # (kernel, {kernel_pars})
    'dt': resolution # dt
}
sig = emb.unfold(to_signal=True, **signal_pars)
batch_time = (signal_pars['duration'] * batch_size) / resolution

# # ####################################################################
# # task mappings and batch preparation
# # ####################################################################
#
# batch_time = (signal_pars['duration'] * batch_size) / resolution
# inputs = np.empty(shape=(int(batch_time), int(n_batches), int(emb.embedding_dimensions)), dtype=np.float32)
# targets = np.empty(shape=(int(batch_time), int(n_batches), int(len(sequencer.tokens))), dtype=np.float32)
#
# for batch in tqdm(range(n_batches), desc="Generating batches"):
#     # inputs
#     batch_input_sequence = sequencer.generate_random_sequence(T=batch_size)
#     # discrete_stim_seq = emb.draw_stimulus_sequence(batch_sequence, as_array=True)
#     batch_inputs = sig.draw_stimulus_sequence(batch_input_sequence, continuous=True)
#     inputs[:, batch, :] = batch_inputs[0].as_array().astype(np.float32).T
#
#     # targets
#     batch_output_seq = sequencer.generate_default_outputs(batch_input_sequence, max_memory=0, max_prediction=0,
#                                        max_chunk=0, chunk_memory=False, chunk_prediction=False)[0]
#     # if len(batch_output_seq) > 1.:
#     #     raise ValueError("Only one task can be assigned at a time")
#     output_signal_pars = (lambda a, b: a.update(b) or a)(signal_pars,{'kernel': ('box', {}), 'amplitude': 1.})
#     out_signal = VectorEmbeddings(vocabulary=np.unique(batch_output_seq['output'])).one_hot().unfold(to_signal=True,
#                                                                     **output_signal_pars)
#     # batch_outputs = VectorEmbeddings(vocabulary=np.unique(batch_output_seq['output'])).one_hot()
#     batch_outputs = out_signal.draw_stimulus_sequence(batch_output_seq['output'], continuous=True)
#     targets[:, batch, :] = batch_outputs[0].as_array().astype(np.float32).T


# ####################################################################
# Network
# ####################################################################
from fna.networks.ann import ContinuousRateRNN
from fna.networks.rnn.helper import prepare_symbolic_batch

# timeconstants, transfer function, weight_parameters
N = 100
taus = np.ones(N) * 100.
noise = np.sqrt(2*(resolution/taus)) * 0.5

net_params = {
    'N': N,
    'transfer_fcn': 'sigmoid', # tanh, relu, sigmoid
    'EI_ratio': 0.8, # set ==0 if no division of E/I is necessary
    'tau_x': taus,
    'tau_r': np.ones(N),
    'noise': noise.astype(np.float32),
    'initial_state': (np.random.uniform, {'low': 0., 'high': 1.}),
    'learning_rate': 2e-2
}
connection_params = {
    'w_in': {
        'density': 1.,
        'distribution': (np.random.uniform, {'low': -1., 'high': 1.})
    }, # np.array(nU,N) or (distribution, {parameters})
    'w_rec': {
        'ei_balance': 5., # wI = ei_balance * wE
        'density': 0.1,
        'distribution': (np.random.normal, {'scale': 0.2, 'loc': 1.})
    },
    'w_out': {
        'density': 1.,
        'distribution': (np.random.uniform, {'low': -1., 'high': 1.})
    }
}

extractor_parameters = {
    'hidden': {
        'population': None,
        'variable': 'active_state',
        'sampling_times': ['stim_offset']}}
decoding_parameters = {
    'reg': {
        'algorithm': "pinv-sgd",
        'extractor': 'hidden',
        'save': True
    },
}
decoder_outputs = {
    'max_memory': 0,
    'max_chunk': 0,
    'max_prediction': 0,
    'chunk_memory': False,
    'chunk_prediction': False}

rnn = ContinuousRateRNN(label='ContinuousRNN', network_parameters=net_params,
                        connection_parameters=connection_params, input_dim=emb.embedding_dimensions,
                        output_dim=len(sequencer.tokens), dt=resolution, extractors=extractor_parameters,
                        decoders=decoding_parameters)

# Train set
train_batch = prepare_symbolic_batch(simulator=rnn.simulator, n_batches=n_batches, batch_size=batch_size,
                                     sequencer=sequencer, continuous_embedding=sig,
                                     batch_time=int(batch_time), as_tensor=True, signal_pars=signal_pars,
                                     decoder_output_pars=decoder_outputs)
train_results = rnn.train(data_batch=train_batch, n_epochs=n_epochs, weight_penalty=0., rate_penalty=0.,
                          clip_max_gradient=0.1, gpu_id=None, symbolic=True, verbose=True, save=False)

# Test set
test_batch = prepare_symbolic_batch(simulator=rnn.simulator,
                                    n_batches=n_batches, batch_size=batch_size,
                                    sequencer=sequencer, continuous_embedding=sig,
                                    batch_time=int(batch_time), as_tensor=True, signal_pars=signal_pars,
                                    decoder_output_pars=decoder_outputs)
test_results, decoder_accuracy = rnn.test(data_batch=test_batch, output_parsing='k-WTA', symbolic=True, save=False)


# train_results = rnn.train(n_epochs=n_epochs, inputs=inputs, targets=targets, weight_penalty=0., rate_penalty=0.,
#           clip_max_gradient=1., gpu_id=None, verbose=True)
#
# test_results = rnn.predict(inputs=inputs, targets=targets)


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(train_results['losses'])

fig1 = plt.figure()
fig1.suptitle("States")
ax1 = fig1.add_subplot(211)
ax2 = fig1.add_subplot(212)

ax1.plot(train_results['states'][0][:, 0, 0])
ax1.plot(train_results['states'][0][:, 0, 10])
ax1.plot(train_results['states'][0][:, 0, 30])

ax2.plot(train_results['states'][-1][:, 0, 0])
ax2.plot(train_results['states'][-1][:, 0, 10])
ax2.plot(train_results['states'][-1][:, 0, 30])


fig2 = plt.figure()
fig2.suptitle("Outputs")
ax21 = fig2.add_subplot(211)
ax22 = fig2.add_subplot(212)
ax21.plot(train_results['outputs'][0][:, 0, 0], '-k', label='output')
ax21.plot(train_batch['targets'][:, 0, 0], '-r', label='target')

ax22.plot(train_results['outputs'][1][:, 0, 0], '-k', label='output')
ax22.plot(train_batch['targets'][:, 0, 0], '-r', label='target')
plt.legend()
plt.show()
