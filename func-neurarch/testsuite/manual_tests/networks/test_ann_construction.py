import tensorflow as tf
import numpy as np

from tasks.symbolic.embeddings import VectorEmbeddings
from tasks.symbolic.sequences import SymbolicSequencer
from tasks.symbolic import NonAdjacentDependencies
# ####################################################################
n_strings = 100
seq_length = 1000

sequencer = SymbolicSequencer(label='random', set_size=10)
emb = VectorEmbeddings(vocabulary=sequencer.tokens).binary_codeword(dim=100, density=0.1)

# ####################################################################
from networks.ann import ArtificialNeuralNetwork

net_params = {
    'N': 128,
    'cell_type': 'LSTM',
    'learning_rate': 0.01
}
ann = ArtificialNeuralNetwork(label='LSTM', network_parameters=net_params, input_dim=emb.embedding_dimensions,
                              output_dim=len(sequencer.tokens))

n_epochs = 100
n_batches = 1200
batch_size = 1000

# Prepare training batches
inputs = []
targets = []
for n in range(n_batches):
    batch_seq = sequencer.generate_random_sequence(T=batch_size)
    input_batch = emb.draw_stimulus_sequence(batch_seq, as_array=True)
    target_seq = sequencer.generate_default_outputs(batch_seq, max_memory=0, max_chunk=0, max_prediction=0)[0]['output']
    target_batch = VectorEmbeddings(vocabulary=sequencer.tokens).one_hot().draw_stimulus_sequence(target_seq,
                                                                                               as_array=True)
    inputs.append(input_batch.T)
    targets.append(target_batch.T)

data_batch = {'inputs': inputs, 'targets': targets}

train_results = ann.train(data_batch, epochs=n_epochs, verbose=True, save=False)

# Prepare test batches
inputs = []
targets = []
for n in range(n_batches):
    batch_seq = sequencer.generate_random_sequence(T=batch_size)
    input_batch = emb.draw_stimulus_sequence(batch_seq, as_array=True)
    target_seq = sequencer.generate_default_outputs(batch_seq, max_memory=0, max_chunk=0, max_prediction=0)[0]['output']
    target_batch = VectorEmbeddings(vocabulary=sequencer.tokens).one_hot().draw_stimulus_sequence(target_seq,
                                                                                                  as_array=True)
    inputs.append(input_batch.T)
    targets.append(target_batch.T)
data_batch = {'inputs': inputs, 'targets': targets}

test_results = ann.predict(data_batch)
