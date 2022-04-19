import tensorflow as tf
import nest
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from fna.tools.parameters import ParameterSet
from fna.tasks.symbolic import SymbolicSequencer
from fna.tasks.symbolic.embeddings import VectorEmbeddings
from fna.networks.rnn.helper import prepare_analog_batch

# ######################################################################################################################
# Continuous integration task
# ######################################################################################################################
data_label = 'analog_processing'
# system parameters
resolution = 1.
net_type = 'ANN'

# task parameters
n_epochs = 2
n_batches = 5
batch_size = 20

sequencer = SymbolicSequencer(label='random sequence', alphabet=['A', 'B'], eos='#')
embedding = VectorEmbeddings(vocabulary=sequencer.tokens).one_hot()
embedding.stimulus_set.update({'#': np.zeros_like(embedding.stimulus_set['A']),
                               'B': -1*embedding.stimulus_set['B']})

# inputs
signal_pars = {
    'duration': 10., # [ms]
    'amplitude': 1., # [max. rate]
    'kernel': ('box', {}),# (kernel, {kernel_pars})
    'dt': resolution # [ms]
}
sig = embedding.unfold(to_signal=True, **signal_pars)
sequencer.generate_stringset(set_length=batch_size, length_range=(2, 2), verbose=False)
batch_seq = sequencer.generate_sequence()
sequencer.string_set = []
batch_time = (signal_pars['duration'] * len(batch_seq)) / resolution


if net_type == 'ANN':
    # ######################################################################################################################
    # 1) VanillaRNN
    # ######################################################################################################################
    from fna.networks.ann import ArtificialNeuralNetwork

    net_params = {
        'N': 500,
        'cell_type': 'LSTM',
        'learning_rate': 2e-2
    }
    rnn = ArtificialNeuralNetwork(label='LSTM', network_parameters=net_params, input_dim=1, output_dim=1)

    # train
    train_batch = prepare_analog_batch(rnn.simulator, n_batches=n_batches, batch_size=batch_size, sequencer=sequencer,
                                   discrete_embedding=embedding, as_tensor=False)

    train_results = rnn.train(train_batch, n_epochs=n_epochs, verbose=True, save=False)

    # variables for analysis
    # loss = train_results['losses']
    # states = train_results['states']
    # outputs = train_results['outputs']
    #
    # batch_time_axis = np.arange(0., len(batch_seq), 1)
    # input_signal = train_batch['inputs'][-1].T
    # state_matrix = states[-1][-1, :, :].T
    # output = outputs[-1][-state_matrix.shape[1]:, :].T
    # target = train_batch['targets'][-1].T

elif net_type == 'rRNN':
    # ######################################################################################################################
    # 3) rRNN
    # ######################################################################################################################
    from fna.networks.rnn import ReservoirRecurrentNetwork

    N = 100
    taus = np.ones(N) * 100.
    noise = np.sqrt(2*(resolution/taus)) * 0.5

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
            'sampling_times': ['stim_offset']}}
    decoding_parameters = {
        'SGD-Regression': {
            'algorithm': "sgd-reg",
            'extractor': 'r_@offset',
            'save': True
        },}

    rnn = ReservoirRecurrentNetwork(label='rRNN-random-amorphous', network_parameters=net_params,
                                    connection_parameters=connection_params, input_dim=1,
                                    extractors=extractor_parameters, decoders=decoding_parameters,
                                    output_dim=1, dt=resolution)
    # Train set
    train_batch = prepare_analog_batch(rnn.simulator, n_batches=n_batches, batch_size=batch_size,
                                       batch_time=int(batch_time), sequencer=sequencer, continuous_embedding=sig,
                                       as_tensor=True)

    train_results = rnn.train(data_batch=train_batch, n_epochs=n_epochs, output_parsing=None, symbolic=False, save=False)

    # variables for analysis
    # loss = [train_results['losses'][epoch]['Batch={}_epoch={}'.format(n_batches, epoch+1)]['{}-integrator'.format(
    #     rnn.decoders.readouts[0].label)]['raw-MSE'] for epoch in range(n_epochs)]
    # states = train_results['states']
    # outputs = train_results['outputs']
    #
    # batch_time_axis = np.arange(0., signal_pars['duration'] * len(batch_seq), resolution)
    # input_signal = train_batch['inputs'][:, -1, :].T
    # state_matrix = states[-1]
    # output = outputs[-1][0]
    # target = train_batch['targets'][:, -1, :].T

elif net_type == 'SNN':
    # ######################################################################################################################
    # 3) SNN
    # ######################################################################################################################
    from examples import example_defaults
    from fna.encoders import NESTEncoder, InputMapper

    example_defaults.reset_kernel(resolution=0.1)
    rnn, rnn_recurrent_connections = example_defaults.default_network(N=100, record_spikes=True)

    encoder = NESTEncoder('inhomogeneous_poisson_generator', label='poisson-input', dim=2)
    # input synapses
    input_synapses = {
        'connect_populations': [('E', encoder.name), ('I', encoder.name), ],
        'weight_matrix': [None, None],
        'conn_specs': [{'rule': 'all_to_all'}, {'rule': 'all_to_all'}],
        'syn_specs': [{'model': 'static_synapse', 'delay': 0.1, 'weight': 1.},
                      {'model': 'static_synapse', 'delay': 0.1, 'weight': 1.}]
    }
    in_to_rnn_connections = InputMapper(source=encoder, target=rnn, parameters=input_synapses)

    # decoder
    extractor_parameters = {
        'E_Vm_@offset': {
            'population': 'E',
            'variable': 'V_m',
            'sampling_times': ['stim_offset']}}
    decoding_parameters = {
        'readout_E_force': {'algorithm': 'force', 'extractor': 'E_Vm_@offset', 'save': True},
        'SGD-Regression': {'algorithm': "sgd-reg", 'extractor': 'E_Vm_@offset', 'save': True},
    }

    # create and connect extractors and decoders
    rnn.connect_state_extractors(extractor_parameters, encoder=encoder, input_mapper=in_to_rnn_connections,
                                 stim_duration=signal_pars['duration'], stim_isi=None, stim_onset=rnn.next_onset_time(),
                                 to_memory=True)
    rnn.create_decoder(decoding_parameters)
    total_delay = in_to_rnn_connections.total_delay + rnn.state_extractor.max_extractor_delay

    # train
    train_batch = prepare_analog_batch(rnn.simulator, n_batches=n_batches, batch_size=batch_size,
                                       batch_time=int(batch_time), sequencer=sequencer, continuous_embedding=sig,
                                       as_tensor=True)
    train_results = rnn.train(train_batch, n_epochs, sig, encoder, rnn.next_onset_time(),
                              total_delay=total_delay, symbolic=False, continuous=True, verbose=True, save=False)

    # prepare variables for analysis
    # loss = [train_results['losses'][epoch]['Batch={}_epoch={}'.format(n_batches, epoch+1)]['{}-classification'.format(
    #     rnn.decoders.readouts[0].label)]['raw-MSE'] for epoch in range(n_epochs)]
    # states = train_results['states']
    # outputs = train_results['outputs']
    #
    # batch_time_axis = np.arange(0., signal_pars['duration'] * batch_size, resolution)
    # input_signal, _ = embedding.draw_stimulus_sequence(train_batch['inputs'][-1], continuous=True, unfold=True,
    #                                                      onset_time=0., verbose=False)
    # input_signal = input_signal.as_array()
    # state_matrix = states[-1][0].matrix
    # output = outputs[-1][0]

    # target_seq = train_batch['decoder_outputs'][-1][0]['output']
    # out_signal = VectorEmbeddings(vocabulary=np.unique(target_seq)).one_hot()
    # target = out_signal.draw_stimulus_sequence(target_seq, as_array=True, verbose=False)
    # target = train_batch['decoder_outputs'][-1][0]['output']

    # stim_onset = 0.1
    # train_batch = prepare_analog_batch(rnn.simulator, n_batches=n_batches, batch_size=batch_size,
    #                                    batch_time=int(batch_time), sequencer=sequencer, continuous_embedding=sig,
    #                                    as_tensor=True)
    #
    # rnn.connect_state_extractors(extractor_parameters, encoder=encoder, input_mapper=in_to_rnn_connections,
    #                              stim_duration=signal_pars['duration'], stim_isi=None, stim_onset=stim_onset,
    #                              to_memory=True)
    # rnn.create_decoder(decoding_parameters)
    # total_delay = in_to_rnn_connections.total_delay + rnn.state_extractor.max_extractor_delay
    # TODO finish this

else:
    raise NotImplementedError("Network type {} not implemented".format(net_type))
"""
# ################################################################################
from decoders.readouts import Readout
from tools.parameters import ParameterSet

# original output (internal)
if hasattr(rnn, "decoders"):
    readout0 = rnn.decoders.readouts[0]
    readout0.predict(state_matrix_test=state_matrix, target_test=target)
else:
    readout_pars = ParameterSet({'algorithm': 'ridge',
                                 'task': 'integration',
                                 'extractor': None})
    readout0 = Readout(label='{}-readout'.format(rnn.name), readout_parameters=readout_pars)
    readout0.output = output
    readout0.test_target = target
perf0 = readout0.evaluate(symbolic=False)
print(perf0)

# external readout
readout_pars = ParameterSet({'algorithm': 'force',
                             'task': 'integration',
                             'extractor': None})
readout1 = Readout(label='{}-readout'.format(rnn.name), readout_parameters=readout_pars)
readout1.train(batch_label='last-train-batch', state_matrix_train=state_matrix, target_train=target)
# readout1.predict(state_matrix_test=state_matrix, target_test=target)
perf1 = readout1.evaluate(symbolic=False)
print(perf1)

# single-sample identity (if continuous)
if net_type in ['cRNN', 'rRNN', 'SNN']:
    single_states = state_matrix[:, ::int(signal_pars['duration']/resolution)]
    single_targets = target[:, ::int(signal_pars['duration']/resolution)]
    readout_pars = ParameterSet({'algorithm': 'pinv',
                                 'task': 'integration',
                                 'extractor': None})
    readout2 = Readout(label='{}-singe-sample-readout'.format(rnn.name), readout_parameters=readout_pars)
    readout2.train(batch_label='single_sample-batch', state_matrix_train=single_states, target_train=single_targets)
    readout2.predict(state_matrix_test=single_states, target_test=single_targets)
    perf2 = readout2.evaluate(symbolic=False)
    print(perf2)
else:
    readout_pars = ParameterSet({'algorithm': 'ridge', #'force',
                                 'task': 'integration',
                                 'extractor': None})
    readout2 = Readout(label='{}-readout'.format(rnn.name), readout_parameters=readout_pars)
    readout2.train(batch_label='single_sample-batch', state_matrix_train=state_matrix, target_train=target)
    readout2.predict(state_matrix_test=state_matrix, target_test=target)
    perf2 = readout2.evaluate(symbolic=False)
    single_targets = target
    print(perf2)

# #################################################################################################
# ANALYZE and PLOT
# #################################################################################################
from tools.visualization.plotting import plot_matrix, plot_trajectory
from tools.analysis.metrics import analyse_state_matrix

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D


analysis_interval = [0., round(batch_time_axis.max())]#[1000., 5000.]
t_idx = [np.where(batch_time_axis == analysis_interval[0])[0][0], np.where(batch_time_axis == analysis_interval[1])[0][0]]
n_neurons = 10 # example neurons to plot

# plot training loss
fig, ax = plt.subplots()
ax.plot(loss)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss")

# Plot sample input signals
input_dimensions = input_signal.shape[0]

fig1, axes = plt.subplots(input_dimensions, 1, sharex=True, figsize=(10, 2*input_dimensions))
if not isinstance(axes, np.ndarray):
    axes = [axes]
for idx, (neuron, ax) in enumerate(zip(range(input_dimensions), axes)):
    ax.plot(batch_time_axis[t_idx[0]:t_idx[1]], input_signal[neuron, t_idx[0]:t_idx[1]], 'k')
    ax.plot(batch_time_axis[t_idx[0]:t_idx[1]], target[neuron, t_idx[0]:t_idx[1]], 'r')
    ax.set_ylabel(r"$U_{"+"{0}".format(neuron)+"}$")
    ax.set_xlim(analysis_interval)
    if idx == input_dimensions - 1:
        ax.set_xlabel("Time [ms]")
# plt.show()

# Plot sample of network states
fig2, axes = plt.subplots(n_neurons, 1, sharex=True, figsize=(10, 2*n_neurons))
neuron_idx = np.random.permutation(state_matrix.shape[0])[:n_neurons]
for idx, (neuron, ax) in enumerate(zip(neuron_idx, axes)):
    ax.plot(batch_time_axis[t_idx[0]:t_idx[1]], state_matrix[neuron, t_idx[0]:t_idx[1]])
    ax.set_ylabel(r"$X_{"+"{0}".format(neuron)+"}$")
    ax.set_xlim(analysis_interval)
    if idx == len(neuron_idx) - 1:
        ax.set_xlabel("Time [ms]")
# plt.show()

# Plot sample trajectory of network states
fig3 = plt.figure()
ax31 = fig3.add_subplot(111, projection='3d')
effective_dimensionality = analyse_state_matrix(state_matrix, stim_labels=None, epochs=None, label=None, plot=False,
                                                display=True, save=False)['dimensionality']
fig3.suptitle(r"$\lambda_{\mathrm{eff}}="+"{}".format(effective_dimensionality)+"$")
plot_trajectory(state_matrix[:, t_idx[0]:t_idx[1]], label="Sample Trajectory", ax=ax31, color='k', display=False, save=False)

# Plot network states matrix
fig4, (ax11, ax12) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [18, 4], 'hspace': 0.1}, sharex=False)
fig4.suptitle("Network States")
_, ax11 = plot_matrix(state_matrix, ax=ax11, save=False, display=False, data_label=None)
ax11.set_ylabel('Neuron')
ax12.plot(batch_time_axis, state_matrix.mean(0), lw=2)
divider2 = make_axes_locatable(ax12)
cax2 = divider2.append_axes("right", size="5%", pad="4%")
cax2.remove()
ax12.set_xlabel("Time [ms]")
ax12.set_xlim([batch_time_axis.min(), batch_time_axis.max()])
ax12.set_ylabel(r"$\bar{X}$")
# plt.show()

# Plot complete outputs and targets (single batch)
n_readouts = 3

fig5, axes = plt.subplots(n_readouts, 1, sharex=False, figsize=(12, 3*n_readouts))
fig5.suptitle("Outputs")

for idx, (r, t) in enumerate(zip([readout0, readout1, readout2], [target, target, single_targets])):
    ax = axes[idx]
    ax.set_title(r'MSE={}'.format(r.performance['raw']['MSE']))
    try:
        ax.plot(r.output[0, :], 'k', label="{0}-{1}".format(r.label, r.algorithm))
        ax.plot(t[0, :], 'r', label='{}-target'.format(r.label))
        ax.set_xlim([0, len(t[0, :])])
    except:
        ax.plot(r.output, 'k', label="{0}-{1}".format(r.label, r.algorithm))
        ax.plot(t, 'r', label='{}-target'.format(r.label))
        ax.set_xlim([0, len(t)])
    ax.legend()
plt.legend()
plt.show()


# # ######################################################################################################################
# readouts = []
# batch_algorithms = ["force", "pinv-sgd", "ridge-sgd", "sgd-reg"]
# tr_ep = 100
#
# for r_alg in batch_algorithms:
#     readout_pars = ParameterSet({'algorithm': r_alg, 'task': 'integration', 'extractor': None})
#     readout01 = Readout(label='{}-readout'.format(rnn.name), readout_parameters=readout_pars)
#
#     # readout01.train(batch_label='sample-batch', state_matrix_train=state_matrix, target_train=target)
#     # readout01.predict(state_matrix_test=state_matrix, target_test=target)
#
#     readout01.train(batch_label='sample-batch', state_matrix_train=single_states, target_train=single_targets)
#     readout01.predict(state_matrix_test=single_states, target_test=single_targets)
#
#     perf01 = readout01.evaluate(symbolic=False)
#     print(perf01)
#     readouts.append(readout01)
#
# n_readouts = len(batch_algorithms)
# fig5, axes = plt.subplots(n_readouts, 1, sharex=False, figsize=(12, 3*n_readouts))
# fig5.suptitle("Outputs")
#
# # for idx, (r, t) in enumerate(zip(readouts, [target for _ in range(len(readouts))])):
# for idx, (r, t) in enumerate(zip(readouts, [single_targets for _ in range(len(readouts))])):
#     ax = axes[idx]
#     ax.set_title(r'MSE={}'.format(r.performance['raw']['MSE']))
#     ax.plot(r.output.ravel(), 'k', label="{0}-{1}".format(r.label, r.algorithm))
#     ax.plot(t.ravel(), 'r', label='{}-target'.format(r.label))
#     ax.set_xlim([0, len(t.ravel())])
#     ax.legend()
# plt.legend()
# plt.show()
"""