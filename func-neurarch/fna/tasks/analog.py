import numpy as np
import collections

from fna.tools import utils

logger = utils.logger.get_logger(__name__)


def narma(input_ts, n=10, narma_pars=None):
    """
    Returns the n-th order non-linear autoregressive moving average (NARMA) of the input time series
    Note: Recommended parameters:
     if n<10 - {'alpha': 0.3, 'beta': 0.05, 'gamma': 1.5, 'eps': 0.1}
     if n>10 - {'alpha': 0.2, 'beta': 0.004, 'gamma': 1.5, 'eps': 0.001}
    :return:
    """
    T = len(input_ts)
    output = np.zeros_like(input_ts)

    if n < 10 and narma_pars is None:
        narma_pars = {'alpha': 0.3, 'beta': 0.05, 'gamma': 1.5, 'eps': 0.1}
    elif n >= 10 and narma_pars is None:
        narma_pars = {'alpha': 0.2, 'beta': 0.004, 'gamma': 1.5, 'eps': 0.001}

    for t in range(n-1, T-1):
        output[t+1] = narma_pars['alpha'] * output[t] + narma_pars['beta'] * output[t] * np.sum(output[t-(n-1):t+1]) + \
                      narma_pars['gamma'] * input_ts[t-(n-1)] * input_ts[t] + narma_pars['eps']
        if n >= 20:
            output[t+1] = np.tanh(output[t+1])
        assert (not np.any(np.isinf(output))), "Unstable solution"
    return output


def mackey_glass(sample_len, tau=17, dt=1., n_samples=1, rng=None):
    """
    Generate the Mackey Glass time-series. Parameters are:
    (adapted from Oger toolbox)
    :param sample_len: length of the time-series in timesteps.
    :param tau: delay of the MG - system. Commonly used values are tau=17 (mild chaos) and tau=30 (moderate chaos). Default is 17.
    :param seed: to seed the random generator, can be used to generate the same timeseries at each invocation
    :param n_samples: number of samples to generate
    :param rng: random number generator state object (optional). Either None or a numpy.random.default_rng object,
        or an object with the same interface
    :return: time_series
    """
    if rng is None:
        rng = np.random.default_rng()

    history_len = int(tau * dt)

    # Initial conditions for the history of the system
    timeseries = 1.2
    samples = []

    for _ in range(n_samples):
        history = collections.deque(1.2 * np.ones(history_len) + 0.2 * (rng.random(history_len) - 0.5))
        # Preallocate the array for the time-series
        inp = np.zeros((sample_len, 1))

        for timestep in range(sample_len):
            for _ in range(int(dt)):
                xtau = history.popleft()
                history.append(timeseries)
                timeseries = history[-1] + (0.2 * xtau / (1.0 + xtau ** 10) - 0.1 * history[-1]) / dt
                inp[timestep] = timeseries
        # squash time series...
        samples.append(inp.T)

    return samples


def flip_flop(p, n_bits, batch_size, n_batches, rng=None):
    """
    Generate data batches for the discrete n_bits flip-flop task
    :param p:
    :param n_bits:
    :param batch_size:
    :param n_batches:
    :param plot:
    :param rng: random number generator state object (optional). Either None or a numpy.random.default_rng object,
        or an object with the same interface
    :return:
    """
    logger.info("Generating discrete {0!s}-bit flip-flop input/output".format(n_bits))

    if rng is None:
        rng = np.random.default_rng()
    batches = []

    for batch in range(n_batches):

        a = rng.binomial(1, p, size=[n_bits, batch_size])
        b = rng.binomial(1, p, size=[n_bits, batch_size])
        inp_ = a-b
        last = 1
        out_ = np.ones_like(inp_)
        for i in range(n_bits):
            for m in range(batch_size):
                a = inp_[i, m]
                if a != 0:
                    last = a
                out_[i, m] = last
        inputs = inp_.T
        outputs = out_.T

        batches.append({
            '{0!s}-bit flip-flop, batch {1!s}'.format(n_bits, batch): {
                'inputs': np.array([inputs]),
                'outputs': np.array([outputs]),
                'accept': [True for _ in range(batch_size)]}})

    return batches
