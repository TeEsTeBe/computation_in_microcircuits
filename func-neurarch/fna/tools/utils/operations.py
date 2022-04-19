"""
========================================================================================================================
 Operations
========================================================================================================================
Common and generic operations performed on various data structures

Functions:
---------------

========================================================================================================================
"""
import itertools
import numpy as np
import pandas as pd


def is_integer(x):
    """
    Check if value is (close to) integer.

    :param x:
    :return:
    """
    return np.isclose(x, np.rint(x))


def empty(signal):
    """
    Evaluate whether a signal is empty
    :param signal:
    :return: bool
    """
    if isinstance(signal, np.ndarray):
        return not bool(signal.size)  # seq.any() # seq.data
    elif isinstance(signal, list) and signal:
        if isiterable(signal):
            result = np.mean([empty(n) for n in list(itertools.chain(signal))])
        else:
            result = np.mean([empty(n) for n in list(itertools.chain(*[signal]))])
        if result == 0. or result == 1.:
            return result.astype(bool)
        else:
            return not result.astype(bool)
    elif isinstance(signal, pd.DataFrame):
        return signal.empty
    else:
        return not signal


def iterate_obj_list(obj_list):
    """
    Build an iterator to iterate through any nested list
    :obj_list: list of objects to iterate
    :return:
    """
    for idx, n in enumerate(obj_list):
        if isinstance(n, list):
            for idxx, nn in enumerate(obj_list[idx]):
                yield obj_list[idx][idxx]
        else:
            yield obj_list[idx]


def reject_outliers(data, m=2.):
    """
    Remove outliers from data
    :param data:
    :param m:
    :return:
    """
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    return data[s < m]


def flatten_list(l):
    """
    Flatten a list containing both iterables (e.g. lists) and non-iterable (int, str) items
    :param l: input list
    :return: flattened_list
    """
    for item in l:
        if isinstance(item, list):
            for subitem in item:
                yield subitem
        else:
            yield item


# def rescale_list(OldList, NewMin, NewMax):
#     NewRange = float(NewMax - NewMin)
#     OldMin = min(OldList)
#     OldMax = max(OldList)
#     OldRange = float(OldMax - OldMin)
#     ScaleFactor = NewRange / OldRange
#     NewList = []
#     for OldValue in OldList:
#         NewValue = ((OldValue - OldMin) * ScaleFactor) + NewMin
#         NewList.append(NewValue)
#     return NewList


def determine_decimal_digits(x):
    """
    Simple function to determine the number of digits after the decimal point
    :param x:
    :return:
    """
    s = str(x)
    if not '.' in s:
        return 0
    return len(s) - s.index('.') - 1


def smooth(x, window_len=11, window='hanning'):
    """
    Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[(window_len // 2 - 1):-(window_len // 2)]


def moving_window(seq, window_len=10):
    """
    Generator for moving window intervals
    :return:
    """
    it = iter(seq)
    result = tuple(itertools.islice(it, window_len))
    if len(result) == window_len:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def isiterable(x):
    """
    Verify if input is iterable (list, dictionary, array...)
    :param x: 	input
    :return: 	boolean
    """
    return hasattr(x, '__iter__') and not isinstance(x, str)


def nesteddict_walk(d, separator='.'):
    """
    Walk a nested dict structure, using a generator

    Nested dictionary entries will be created by joining to the parent dict entry
    separated by separator

    :param d: dictionarypars = ParameterSpace(params_file_full_path)
    :param separator:
    :return: generator object (iterator)
    """

    for key1, value1 in list(d.items()):
        if isinstance(value1, dict):
            for key2, value2 in nesteddict_walk(value1, separator):  # recurse into subdict
                yield "%s%s%s" % (key1, separator, key2), value2
        else:
            yield key1, value1


def contains_instance(search_instances, cls):
    """
    Check whether any of the search instances is in cls.

    :param search_instances: the instance to search for
    :param cls:
    :return: boolean
    """
    return any(isinstance(o, cls) for o in search_instances)


def nesteddict_flatten(d, separator='.'):
    """
    Return a flattened version of a nested dict structure.
    Composite keys are created by joining each key to the key of the parent dict using `separator`.

    :param d: dictionary to flatten
    :param separator:
    :return: flattened dictionary (no nesting)
    """
    flatd = {}
    for k, v in nesteddict_walk(d, separator):
        flatd[k] = v

    return flatd


def copy_dict(source_dict, diffs={}):
    """
    Returns a copy of source dict, updated with the new key-value pairs provided
    :param source_dict: dictionary to be copied and updated
    :param diffs: new key-value pairs to add
    :return: copied and updated dictionary
    """
    assert isinstance(source_dict, dict), "Input to this function must be a dictionary"
    result = source_dict.copy()
    result.update(diffs)
    return result


def delete_keys_from_dict(dict_del, the_keys):
    """
    Delete the keys present in the lst_keys from the dictionary. Loops recursively over nested dictionaries.

    :param dict_del:
    :param the_keys:
    :return: dictionary without the deleted elements
    """
    # make sure the_keys is a set to get O(1) lookups
    if type(the_keys) is not set:
        the_keys = set(the_keys)
    for k,v in list(dict_del.items()):
        if k in the_keys:
            del dict_del[k]
        if isinstance(v, dict):
            delete_keys_from_dict(v, the_keys)
    return dict_del


def clean_array(x):
    """
    Remove None entries from an array and replace with np.nan
    :return:
    """
    for idx, val in np.ndenumerate(x):
        if val is None:
            x[idx] = np.nan
        elif empty(val):
            x[idx] = np.nan
    return x


def is_binary(array):
    """
    Determine if array is binary (all values in {0, 1})
    :param array:
    :return: bool
    """
    return np.array_equal(array, array.astype(bool))
