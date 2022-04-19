from tqdm import tqdm
import numpy as np
import types

from fna.tasks.symbolic.embeddings import VectorEmbeddings
from fna.tasks.preprocessing import ImageFrontend


# ######################################################################################################################
def prepare_symbolic_batch(simulator, n_batches, batch_size, sequencer, discrete_embedding=None,
                           continuous_embedding=None, batch_time=None, as_tensor=False, signal_pars=None,
                           decoder_output_pars=None):
    """
    Generates data batches for symbolic processing tasks
    :param n_batches: Number of unique batches
    :param batch_size: Size of batch (discrete steps or time)
    """
    # stimulus_timings = []
    if decoder_output_pars is None:
        decoder_output_pars = {}

    if simulator == 'Tensorflow' or simulator == 'TensorFlow' or simulator == 'TF' or simulator == 'Python':
        if not as_tensor:
            inputs = []
            targets = []
        else:
            inputs = np.empty(shape=(batch_time, int(n_batches), int(continuous_embedding.embedding_dimensions)),
                              dtype=np.float32)
            targets = np.empty(shape=(batch_time, int(n_batches), int(len(sequencer.tokens))), dtype=np.float32)

        decoder_targets = []
        for batch in tqdm(range(n_batches), desc="Generating batches"):

            # input sequence
            batch_seq = sequencer.generate_random_sequence(T=batch_size, verbose=False)  # TODO modify (provide function)!!
            input_times = np.arange(len(batch_seq))

            # decoder outputs
            batch_targets = sequencer.generate_default_outputs(batch_seq, verbose=False, **decoder_output_pars)
            decoder_targets.append(batch_targets)

            # network inputs
            if discrete_embedding is not None and not as_tensor:
                input_batch = discrete_embedding.draw_stimulus_sequence(batch_seq, as_array=True, verbose=False)
                inputs.append(input_batch.T)

            elif continuous_embedding is not None and as_tensor:
                if isinstance(continuous_embedding, ImageFrontend):
                    input_batch, input_times = continuous_embedding.draw_stimulus_sequence(
                        batch_seq, continuous=True, unfold=True, onset_time=0., verbose=False)
                else:
                    input_batch, input_times = continuous_embedding.draw_stimulus_sequence(
                        batch_seq, continuous=True, onset_time=0., verbose=False)

                if isinstance(input_batch, types.GeneratorType):
                    for t, inp in enumerate(input_batch):
                        inputs[int(inp[1][0]/inp[0].dt):int(inp[1][1]/inp[0].dt), batch, :] = inp[0].as_array().T
                else:
                    inputs[:, batch, :] = input_batch.as_array().astype(np.float32).T

            # network targets
            target_seq = sequencer.generate_default_outputs(batch_seq, max_memory=0, max_chunk=0, max_prediction=0)[0][
                'output']  # classification target for main network output (in ANNs)
            if discrete_embedding is not None:
                target_batch = VectorEmbeddings(vocabulary=sequencer.tokens).one_hot().draw_stimulus_sequence(
                    target_seq, as_array=True, verbose=False)
                targets.append(target_batch.T)
            elif continuous_embedding is not None:
                output_signal_pars = (lambda a, b: a.update(b) or a)(signal_pars, {'kernel': ('box', {}),
                                                                                   'amplitude': 1.})
                out_signal = VectorEmbeddings(vocabulary=np.unique(target_seq)).one_hot().unfold(to_signal=True,
                                                                                                 verbose=False,
                                                                                                 **output_signal_pars)
                target_batch = out_signal.draw_stimulus_sequence(target_seq, continuous=True, verbose=False)
                targets[:, batch, :] = target_batch[0].as_array().astype(np.float32).T

        return {'inputs': inputs, 'targets': targets, 'decoder_outputs': decoder_targets, 'input_times': input_times}

    elif simulator == 'NEST':
        batch_sequence = [sequencer.generate_random_sequence(T=batch_size, verbose=False) for _ in range(n_batches)]
        batch_targets = [sequencer.generate_default_outputs(batch_seq, verbose=False, **decoder_output_pars)
                         for batch_seq in batch_sequence]

        return {'inputs': batch_sequence, 'decoder_outputs': batch_targets}


# def prepare_ann_batch(n_batches, batch_size, sequencer, embedder):
#     """
#     Generate data batches for discrete ANN
#     :param n_batches: Number of batches
#     :param batch_size: Size of each data batch
#     :param sequencer: sequencer object
#     :param embedder: embedding object
#     :return:
#     """
#     # Prepare training batches
#     inputs = []
#     targets = []
#     for n in range(n_batches):
#         batch_seq = sequencer.generate_random_sequence(T=batch_size, verbose=False)
#         input_batch = embedder.draw_stimulus_sequence(batch_seq, as_array=True)
#         target_seq = sequencer.generate_default_outputs(batch_seq, max_memory=0, max_chunk=0, max_prediction=0)[0]['output']
#         target_batch = VectorEmbeddings(vocabulary=sequencer.tokens).one_hot().draw_stimulus_sequence(target_seq,
#                                                                                                       as_array=True)
#         inputs.append(input_batch.T)
#         targets.append(target_batch.T)
#
#     data_batch = {'inputs': inputs, 'targets': targets}
#
#     return data_batch


def prepare_analog_batch(simulator, n_batches, batch_size, sequencer, discrete_embedding=None,
                         continuous_embedding=None, batch_time=None, as_tensor=False, signal_pars=None,
                         decoder_output_pars=None):
    """
    Generates data batches for analog processing tasks
    :param n_batches: Number of unique batches
    :param batch_size: Size of batch (discrete steps or time)
    """
    if simulator == 'Tensorflow' or simulator == 'TensorFlow' or simulator == 'TF' or simulator == 'Python':
        decoder_targets = []
        if not as_tensor:
            inputs = []
            targets = []
        else:
            inputs = np.empty(shape=(batch_time, int(n_batches), int(1)),
                              dtype=np.float32)
            targets = np.empty(shape=(batch_time, int(n_batches), int(1)), dtype=np.float32)

        for batch in tqdm(range(n_batches), desc="Generating batches"):
            # inputs
            sequencer.generate_stringset(set_length=batch_size, length_range=(2, 2), verbose=False)
            batch_seq = sequencer.generate_sequence()
            sequencer.string_set = []

            if discrete_embedding is not None:
                signal = discrete_embedding.draw_stimulus_sequence(batch_seq, as_array=True, verbose=False).sum(axis=0)
                input_times = []

            if continuous_embedding is not None:
                signal, input_times = continuous_embedding.draw_stimulus_sequence(batch_seq, onset_time=0., continuous=True,
                                                                   intervals=None, verbose=False)
                signal = signal.as_array().sum(axis=0)
            input_batch = np.reshape(signal, (1, signal.shape[0]))
            target_batch = np.reshape(np.cumsum(signal), (1, signal.shape[0]))
            target_batch = target_batch / (target_batch.max() - target_batch.min())

            if continuous_embedding is not None:
                decoder_targets.append([{
                    'label': 'integrator',
                    'output': target_batch.ravel()[::int(continuous_embedding.stimulus_set['A'][0].signal_length)],
                    'accept': [True for _ in range(target_batch.shape[1])][::int(continuous_embedding.stimulus_set['A'][0].signal_length)]}])
            else:
                decoder_targets.append([{
                    'label': 'integrator',
                    'output': target_batch,
                    'accept': [True for _ in range(target_batch.shape[1])]}])

            if not as_tensor:
                inputs.append(input_batch.T)
                targets.append(target_batch.T)
            else:
                inputs[:, batch, :] = input_batch.astype(np.float32).T
                targets[:, batch, :] = target_batch.astype(np.float32).T

        return {'inputs': inputs, 'targets': targets, 'decoder_outputs': decoder_targets, 'input_times': input_times}

    elif simulator == 'NEST':
        decoder_targets = []
        batch_sequence = []
        for _ in range(n_batches):
            sequencer.generate_stringset(set_length=batch_size, length_range=(2, 2), verbose=False)
            batch_seq = sequencer.generate_sequence()
            batch_sequence.append(batch_seq)
            sequencer.string_set = []
            if discrete_embedding is not None:
                signal = discrete_embedding.draw_stimulus_sequence(batch_seq, as_array=True, verbose=False).sum(axis=0)
                input_times = []
            if continuous_embedding is not None:
                signal, input_times = continuous_embedding.draw_stimulus_sequence(batch_seq, onset_time=0., continuous=True,
                                                                                  intervals=None, verbose=False)
                signal = signal.as_array().sum(axis=0)
            input_batch = np.reshape(signal, (1, signal.shape[0]))
            target_batch = np.reshape(np.cumsum(signal), (1, signal.shape[0]))
            target_batch = target_batch / (target_batch.max() - target_batch.min())

            decoder_targets.append([{
                        'label': 'integrator',
                        'output': target_batch.ravel()[::int(continuous_embedding.stimulus_set['A'][0].signal_length)],
                        'accept': [True for _ in range(target_batch.shape[1])][::int(continuous_embedding.stimulus_set['A'][0].signal_length)]}])

            # batch_targets = [sequencer.generate_default_outputs(batch_seq, verbose=False, **decoder_output_pars)
            #                  for batch_seq in batch_sequence]

        return {'inputs': batch_sequence, 'decoder_outputs': decoder_targets,}