import re
import numpy as np
from matplotlib import pyplot as pl

from fna import tools
from fna.tools.signals.spikes import SpikeTrain, SpikeList


class AnalogSignal(object):
    """
    AnalogSignal(signal, dt, t_start=0, t_stop=None)

    Return a AnalogSignal object which will be an analog signal trace

    Inputs:
        signal  - the vector with the data of the AnalogSignal
        dt      - the time step between two data points of the sampled analog signal
        t_start - beginning of the signal, in ms.
        t_stop  - end of the SignalList, in ms. If None, will be inferred from the data

    Examples:
        >> s = AnalogSignal(range(100), dt=0.1, t_start=0, t_stop=10)

    See also
        AnalogSignalList
    """
    def __init__(self, signal, dt, t_start=None, t_stop=None):
        self.signal = np.array(signal, float)
        self.dt = float(dt)
        if t_start is None:
            t_start = 0
        self.t_start = float(t_start)
        self.t_stop = float(t_stop)
        self.signal_length = len(signal)
        self.closed_time_interval = False # to allow signals that include the last time step or not.. (dirty fix)
        if self.t_stop is not None:
            t_axis = np.arange(self.t_start, self.t_stop + 0.0001, self.dt)
            if len(self.signal) == len(t_axis):
                self.closed_time_interval = True
            elif len(self.signal) == len(t_axis[:-1]):
                self.closed_time_interval = False
            else:
                raise Exception("Inconsistent arguments: t_start=%g, t_stop=%g, dt=%g implies %d elements, actually %d"
                                % (self.t_start, self.t_stop, self.dt,
                                   int(round((self.t_stop - self.t_start) / float(self.dt))), len(self.signal)))
        else:
            self.t_stop = round(self.t_start + len(self.signal) * self.dt, 2)

        if self.t_start >= self.t_stop:
            raise Exception("Incompatible time interval for the creation of the AnalogSignal. t_start=%s, t_stop=%s" %
                            (self.t_start, self.t_stop))

    def __getslice__(self, i, j):
        """
        Return a sublist of the signal vector of the AnalogSignal
        """
        return self.signal[i:j]

    def raw_data(self):
        """
        Return the signal
        """
        return self.signal

    def duration(self):
        """
        Return the duration of the SpikeTrain
        """
        return self.t_stop - self.t_start

    def __str__(self):
        return str(self.signal)

    def __len__(self):
        return len(self.signal)

    def max(self):
        return self.signal.max()

    def min(self):
        return self.signal.min()

    def mean(self):
        return np.mean(self.signal)

    def copy(self):
        """
        Return a copy of the AnalogSignal object
        """
        return AnalogSignal(self.signal, self.dt, self.t_start, self.t_stop)

    def time_axis(self, normalized=False):
        """
        Return the time axis of the AnalogSignal
        """
        if normalized:
            norm = self.t_start
        else:
            norm = 0.
        if self.closed_time_interval:
            return np.arange(self.t_start - norm, self.t_stop - norm + 0.0001, self.dt)
        else:
            return np.arange(self.t_start - norm, self.t_stop - norm, self.dt)

    def time_offset(self, offset):
        """
        Add a time offset to the AnalogSignal object. t_start and t_stop are
        shifted from offset.

        Inputs:
            offset - the time offset, in ms

        Examples:
            >> as = AnalogSignal(arange(0,100,0.1),0.1)
            >> as.t_stop
                100
            >> as.time_offset(1000)
            >> as.t_stop
                1100
        """
        t_start = self.t_start + offset
        t_stop = self.t_stop + offset
        return AnalogSignal(self.signal, self.dt, t_start, t_stop)

    def time_parameters(self):
        """
        Return the time parameters of the AnalogSignal (t_start, t_stop, dt)
        """
        return (self.t_start, self.t_stop, self.dt)

    def time_slice(self, t_start, t_stop):
        """
        Return a new AnalogSignal obtained by slicing between t_start and t_stop

        Inputs:
            t_start - begining of the new SpikeTrain, in ms.
            t_stop  - end of the new SpikeTrain, in ms.

        See also:
            interval_slice
        """
        assert t_start >= self.t_start
        # assert t_stop <= self.t_stop
        assert t_stop > t_start

        t = self.time_axis()
        i_start = int(round((t_start - self.t_start) / self.dt))
        i_stop = int(round((t_stop - self.t_start) / self.dt))
        signal = self.signal[i_start:i_stop]
        result = AnalogSignal(signal, self.dt, t_start, t_stop)
        return result

    def interval_slice(self, interval):
        """
        Return only the parts of the AnalogSignal that are defined in the range of the interval.
        The output is therefor a list of signal segments

        Inputs:
            interval - The Interval to slice the AnalogSignal with

        Examples:
            >> as.interval_slice(Interval([0,100],[50,150]))

        See also:
            time_slice
        """
        result = []
        for itv in interval.interval_data:
            result.append(self.signal[itv[0] // self.dt:itv[1] // self.dt])
        return result

    def event_triggered_average(self, events, average=True, t_min=0, t_max=100, with_time=False):
        """
        Return the spike triggered averaged of an analog signal according to selected events,
        on a time window t_spikes - tmin, t_spikes + tmax
        Can return either the averaged waveform (average = True), or an array of all the
        waveforms triggered by all the spikes.

        Inputs:
            events  - Can be a SpikeTrain object (and events will be the spikes) or just a list
                      of times
            average - If True, return a single vector of the averaged waveform. If False,
                      return an array of all the waveforms.
            t_min   - Time (>0) to average the signal before an event, in ms (default 0)
            t_max   - Time (>0) to average the signal after an event, in ms  (default 100)

        Examples:
            >> vm.event_triggered_average(spktrain, average=False, t_min = 50, t_max = 150)
            >> vm.event_triggered_average(spktrain, average=True)
            >> vm.event_triggered_average(range(0,1000,10), average=False)
        """

        if isinstance(events, SpikeTrain):
            events = events.spike_times
        else:
            assert np.iterable(events), "events should be a SpikeTrain object or an iterable object"
        assert (t_min >= 0) and (t_max >= 0), "t_min and t_max should be greater than 0"
        assert len(events) > 0, "events should not be empty and should contained at least one element"

        time_axis = np.linspace(-t_min, t_max, (t_min + t_max) / self.dt)
        N = len(time_axis)
        Nspikes = 0.
        if average:
            result = np.zeros(N, float)
        else:
            result = []

        # recalculate everything into timesteps, is more stable against rounding errors
        #  and subsequent cutouts with different sizes
        events = np.floor(np.array(events) / self.dt)
        t_min_l = np.floor(t_min / self.dt)
        t_max_l = np.floor(t_max / self.dt)
        t_start = np.floor(self.t_start / self.dt)
        t_stop = np.floor(self.t_stop / self.dt)

        for spike in events:
            if ((spike - t_min_l) >= t_start) and ((spike + t_max_l) < t_stop):
                spike = spike - t_start
                if average:
                    result += self.signal[(spike - t_min_l):(spike + t_max_l)]
                else:
                    result.append(self.signal[(spike - t_min_l):(spike + t_max_l)])
                Nspikes += 1
        if average:
            result = result / Nspikes
        else:
            result = np.array(result)

        if with_time:
            return result, time_axis
        else:
            return result

    def slice_by_events(self, events, t_min=100, t_max=100):
        """
        Returns a dict containing new AnalogSignals cutout around events.

        Inputs:
            events  - Can be a SpikeTrain object (and events will be the spikes) or just a list
                      of times
            t_min   - Time (>0) to cut the signal before an event, in ms (default 100)
            t_max   - Time (>0) to cut the signal after an event, in ms  (default 100)

        Examples:
            >> res = aslist.slice_by_events([100,200,300], t_min=0, t_max =100)
            >> print len(res)
                3
        """
        if isinstance(events, SpikeTrain):
            events = events.spike_times
        else:
            assert np.iterable(events), "events should be a SpikeTrain object or an iterable object"
        assert (t_min >= 0) and (t_max >= 0), "t_min and t_max should be greater than 0"
        assert len(events) > 0, "events should not be empty and should contained at least one element"

        result = {}
        for index, spike in enumerate(events):
            if ((spike - t_min) >= self.t_start) and ((spike + t_max) < self.t_stop):
                spike = spike - self.t_start
                t_start_new = (spike-t_min)
                t_stop_new = (spike+t_max)
                result[index] = self.time_slice(t_start_new, t_stop_new)
        return result

    def slice_exclude_events(self,events,t_min=100,t_max=100):
        """
        yields new AnalogSignals with events cutout (useful for removing spikes).

        Events should be sorted in chronological order

        Inputs:
            events  - Can be a SpikeTrain object (and events will be the spikes) or just a list
                      of times
            t_min   - Time (>0) to cut the signal before an event, in ms (default 100)
            t_max   - Time (>0) to cut the signal after an event, in ms  (default 100)

        Examples:
            >> res = aslist.slice_by_events([100,200,300], t_min=0, t_max =10)
            >> print len(res)
                4

        Author: Eilif Muller
        """
        if isinstance(events, SpikeTrain):
            events = events.spike_times
        else:
            assert np.iterable(events), "events should be a SpikeTrain object or an iterable object"
        assert (t_min >= 0) and (t_max >= 0), "t_min and t_max should be greater than 0"

        # if no events to remove, return self
        if len(events) == 0:
            yield self
            return

        t_last = self.t_start
        for spike in events:
            # skip spikes which aren't close to the signal interval
            if spike + t_min < self.t_start or spike - t_min > self.t_stop:
                continue

            t_min_local = np.max([t_last, spike - t_min])
            t_max_local = np.min([self.t_stop, spike + t_max])

            if t_last < t_min_local:
                yield self.time_slice(t_last, t_min_local)

            t_last = t_max_local

        if t_last < self.t_stop:
            yield self.time_slice(t_last, self.t_stop)

    def cov(self, signal):
        """

        Returns the covariance of two signals (self, signal),

        i.e. mean(self.signal*signal)-mean(self.signal)*(mean(signal)


        Inputs:
            signal  - Another AnalogSignal object.  It should have the same temporal dimension
                      and dt.

        Examples:
            >> a1 = AnalogSignal(numpy.random.normal(size=1000),dt=0.1)
            >> a2 = AnalogSignal(numpy.random.normal(size=1000),dt=0.1)
            >> print a1.cov(a2)
            -0.043763817072107143
            >> print a1.cov(a1)
            1.0063757246782141

        See also:
            NeuroTools.analysis.ccf
            http://en.wikipedia.org/wiki/Covariance

        Author: Eilif Muller

        """

        assert signal.dt == self.dt
        assert signal.signal.shape == self.signal.shape

        return np.mean(self.signal * signal.signal) - np.mean(self.signal) * (np.mean(signal.signal))


class AnalogSignalList(object):
    """
    AnalogSignalList(signals, id_list, times=None, dt=None, t_start=None, t_stop=None, dims=None)

    Return a AnalogSignalList object which will be a list of AnalogSignal objects.

    Inputs:
        signals - a list of tuples (id, value) with all the values sorted in time of the analog signals
        id_list - the list of the ids of all recorded cells (needed for silent cells)
        times   - array of sampled time points (if dt is None, it will be inferred from times)
        dt      - if dt is specified, time values should be floats
        t_start - beginning of the List, in ms.
        t_stop  - end of the List, in ms. If None, will be inferred from the data
        dims    - dimensions of the recorded population, if not 1D population

    dt, t_start and t_stop are shared for all SpikeTrains object within the SpikeList

    See also
        load_currentlist load_vmlist, load_conductancelist
    """
    def __init__(self, signals, id_list, dt=None, times=None, t_start=None, t_stop=None, dims=None):

        if dt is None:
            assert times is not None, "dt or times must be specified"
            dt = np.mean(np.diff(np.unique(times)))
        self.dt = np.round(float(dt), tools.utils.operations.determine_decimal_digits(dt))

        if t_start is None and times is None:
            t_start = 0.
        elif t_start is None and times is not None:
            t_start = min(times)
            if abs(np.round(min(times) - np.round(min(times)), tools.utils.operations.determine_decimal_digits(self.dt))) <= self.dt:
                    t_start = np.round(min(times), tools.utils.operations.determine_decimal_digits(self.dt))
        self.t_start = t_start
        #
        if t_stop is None and times is None:
            t_stop = len(signals) * self.dt
        elif t_stop is None and times is not None:
            t_stop = max(times)
            if abs(np.round(max(times) - np.round(max(times)), tools.utils.operations.determine_decimal_digits(self.dt))) <= self.dt:
                t_stop = np.round(max(times), tools.utils.operations.determine_decimal_digits(self.dt))
                if t_stop == max(times) and self.dt >= 1.:
                    t_stop += self.dt
        self.t_stop = round(t_stop, tools.utils.operations.determine_decimal_digits(self.dt))

        self.dimensions = dims
        self.analog_signals = {}
        self.signal_length = len(signals)

        signals = np.array(signals)

        signal_ids = [np.transpose(signals[signals[:, 0] == id, :])[1] for id in id_list]
        lens = np.array(list(map(len, signal_ids)))

        for id, signal in zip(id_list, signal_ids):
            self.organize_analogs(signal, id, lens) # trying to speed up...
        signals = list(self.analog_signals.values())
        if signals:
            self.signal_length = len(signals[0])
            for signal in signals:
                if len(signal) != self.signal_length:
                    raise Exception("Signals must all be the same length %d != %d" % (self.signal_length, len(signal)))

        if t_stop is None:
            self.t_stop = self.t_start + self.signal_length * self.dt

    # TODO comment
    def organize_analogs(self, signal, id, lens):
        if len(signal) > 0 and len(signal) == max(lens):
            self.analog_signals[id] = AnalogSignal(signal, self.dt, self.t_start, self.t_stop)
        elif len(signal) > 0 and len(signal) != max(lens):
            sig = np.zeros(max(lens))
            sig[:len(signal)] = signal.copy()
            steps_left = max(lens) - len(signal)
            sig[len(signal):] = np.ones(steps_left) * signal[-1]
            self.analog_signals[id] = AnalogSignal(sig, self.dt, self.t_start, self.t_stop)

    def id_list(self):
        """
        Return the list of all the cells ids contained in the
        SpikeList object
        """
        return np.sort(np.array(list(self.analog_signals.keys())))

    def copy(self):
        """
        Return a copy of the AnalogSignalList object
        """
        aslist = AnalogSignalList([], [], self.dt, self.t_start, self.t_stop, self.dimensions)
        for id in self.id_list():
            aslist.append(id, self.analog_signals[id])
        return aslist

    # TODO
    # def merge(self, signals):
    #     """
    #     For each cell id in signals (AnalogSignalList) that matches an id in this AnalogSignalList, merge the two
    #     AnalogSignals and save the result in this AnalogSignalList.
    #     Note that AnalogSignals with ids not in this AnalogSignalList are appended to it.
    #     :param signals: AnalogSignalList to be merged with current one
    #     :return:
    #     """
    #     for id, analog_signal in signals.analog_signals.items():
    #         if id in self.id_list():
    #             self.analog_signals[id].merge(analog_signal)
    #         else:
    #             self.append(id, analog_signal)

    def __getitem__(self, id):
        if id in self.id_list():
            return self.analog_signals[id]
        else:
            raise Exception("id %d is not present in the AnalogSignal. See id_list()" %id)

    def __setitem__(self, i, val):
        assert isinstance(val, AnalogSignal), "An AnalogSignalList object can only contain AnalogSignal objects"
        if len(self) > 0:
            errmsgs = []
            for attr in "dt", "t_start", "t_stop":
                if getattr(self, attr) == 0:
                    if getattr(val, attr) != 0:
                        errmsgs.append("%s: %g != %g (diff=%g)" % (attr, getattr(val, attr), getattr(self, attr),
                                                                   getattr(val, attr) - getattr(self, attr)))
                elif (getattr(val, attr) - getattr(self, attr))/getattr(self, attr) > 1e-12:
                    errmsgs.append("%s: %g != %g (diff=%g)" % (attr, getattr(val, attr), getattr(self, attr),
                                                               getattr(val, attr)-getattr(self, attr)))
            if len(val) != self.signal_length:
                errmsgs.append("signal length: %g != %g" % (len(val), self.signal_length))
            if errmsgs:
                raise Exception("AnalogSignal being added does not match the existing signals: "+", ".join(errmsgs))
        else:
            self.signal_length = len(val)
            self.t_start = val.t_start
            self.t_stop = val.t_stop
        self.analog_signals[i] = val

    def __len__(self):
        return len(self.analog_signals)

    def __iter__(self):
        return iter(self.analog_signals.values())

    def __sub_id_list(self, sub_list=None):
        if sub_list == None:
            return self.id_list()
        if type(sub_list) == int:
            return np.random.permutation(self.id_list())[0:sub_list]
        if type(sub_list) == list:
            return sub_list

    def append(self, id, signal):
        """
        Add an AnalogSignal object to the AnalogSignalList

        Inputs:
            id     - the id of the new cell
            signal - the AnalogSignal object representing the new cell

        The AnalogSignal object is sliced according to the t_start and t_stop times
        of the AnalogSignallist object

        See also
            __setitem__
        """
        assert isinstance(signal, AnalogSignal), "An AnalogSignalList object can only contain AnalogSignal objects"
        if id in self.id_list():
            raise Exception("Id already present in AnalogSignalList. Use setitem instead()")
        else:
            self[id] = signal

    def time_axis(self):
        """
        Return the time axis of the AnalogSignalList object
        """
        if all(self.analog_signals[x].closed_time_interval for x in self.analog_signals):
            return np.arange(self.t_start, self.t_stop + 0.00001, self.dt) # !! (adding 0.00001 to have the last timestep
        else:
            return np.arange(self.t_start, self.t_stop, self.dt)

    def id_offset(self, offset):
        """
        Add an offset to the whole AnalogSignalList object. All the id are shifted
        with a offset value.

        Inputs:
            offset - the id offset

        Examples:
            >> as.id_list()
                [0,1,2,3,4]
            >> as.id_offset(10)
            >> as.id_list()
                [10,11,12,13,14]
        """
        id_list = np.sort(self.id_list())
        N = len(id_list)
        for idx in range(1, len(id_list) + 1):
            id = id_list[N - idx]
            spk = self.analog_signals.pop(id)
            self.analog_signals[id + offset] = spk

    def id_slice(self, id_list):
        """
        Return a new AnalogSignalList obtained by selecting particular ids

        Inputs:
            id_list - Can be an integer (and then N random cells will be selected)
                      or a sublist of the current ids

        The new AnalogSignalList inherits the time parameters (t_start, t_stop, dt)

        See also
            time_slice
        """
        new_AnalogSignalList = AnalogSignalList([], [], dt=self.dt, t_start=self.t_start, t_stop=self.t_stop,
                                                dims=self.dimensions)
        id_list = self.__sub_id_list(id_list)
        for id in id_list:
            try:
                new_AnalogSignalList.append(id, self.analog_signals[id])
            except Exception:
                print("id %d is not in the source AnalogSignalList" %id)
        return new_AnalogSignalList

    def time_offset(self, offset):
        """
        Shifts the time axis by offset
        :param offset:
        :return:
        """
        new_AnalogSignalList = AnalogSignalList([], [], dt=self.dt, t_start=self.t_start+offset,
                                                t_stop=self.t_stop+offset, dims=self.dimensions)
        for id in self.id_list():
            an_signal = self.analog_signals[id].copy()
            new_an_signal = an_signal.time_offset(offset)
            new_AnalogSignalList.append(id, new_an_signal)

        return new_AnalogSignalList

    def time_slice(self, t_start, t_stop):
        """
        Return a new AnalogSignalList obtained by slicing between t_start and t_stop

        Inputs:
            t_start - begining of the new AnalogSignalList, in ms.
            t_stop  - end of the new AnalogSignalList, in ms.

        See also
            id_slice
        """
        new_AnalogSignalList = AnalogSignalList([], [], dt=self.dt, t_start=t_start, t_stop=t_stop,
                                                dims=self.dimensions)
        for id in self.id_list():
            new_AnalogSignalList.append(id, self.analog_signals[id].time_slice(t_start, t_stop))
        return new_AnalogSignalList

    def select_ids(self, criteria=None):
        """
        Return the list of all the cells in the AnalogSignalList that will match the criteria
        expressed with the following syntax.

        Inputs :
            criteria - a string that can be evaluated on a AnalogSignal object, where the
                       AnalogSignal should be named ``cell''.

        Exemples:
            >> aslist.select_ids("mean(cell.signal) > 20")
            >> aslist.select_ids("cell.std() < 0.2")
        """
        selected_ids = []
        for id in self.id_list():
            cell = self.analog_signals[id]
            if eval(criteria):
                selected_ids.append(id)
        return selected_ids

    def convert(self, format="[values, ids]"):
        """
        Return a new representation of the AnalogSignalList object, in a user designed format.
            format is an expression containing either the keywords values and ids,
            time and id.

        Inputs:
            format    - A template to generate the corresponding data representation, with the keywords
                        values and ids

        Examples:
            >> aslist.convert("[values, ids]") will return a list of two elements, the
                first one being the array of all the values, the second the array of all the
                corresponding ids, sorted by time
            >> aslist.convert("[(value,id)]") will return a list of tuples (value, id)
        """
        is_values = re.compile("values")
        is_ids = re.compile("ids")
        values = np.concatenate([st.signal for st in self.analog_signals.values()])
        ids = np.concatenate([id * np.ones(len(st.signal), int) for id, st in self.analog_signals.items()])
        if is_values.search(format):
            if is_ids.search(format):
                return eval(format)
            else:
                raise Exception("You must have a format with [values, ids] or [value, id]")
        is_values = re.compile("value")
        is_ids = re.compile("id")
        if is_values.search(format):
            if is_ids.search(format):
                result = []
                for id, time in zip(ids, values):
                    result.append(eval(format))
            else:
                raise Exception("You must have a format with [values, ids] or [value, id]")
            return result

    def raw_data(self):
        """
        Function to return a N by 2 array of all values and ids.

        Examples:
            >> spklist.raw_data()
            >> array([[  1.00000000e+00,   1.00000000e+00],
                      [  1.00000000e+00,   1.00000000e+00],
                      [  2.00000000e+00,   2.00000000e+00],
                         ...,
                      [  2.71530000e+03,   2.76210000e+03]])

        See also:
            convert()
        """
        data = np.array(self.convert("[values, ids]"))
        data = np.transpose(data)
        return data

    def as_array(self):
        """
        Return the analog signal list as an array (len(id_list) x len(time_axis))
        """
        if len(self.analog_signals[self.id_list()[0]].raw_data()) != len(self.time_axis()):
            time_axis = self.time_axis()[:-1] # in some cases, the time is rounded and causes and error
        else:
            time_axis = self.time_axis()

        a = np.zeros((len(self.id_list()), len(time_axis)))
        # print len(self.time_axis())
        for idx, n in enumerate(self.id_list()):
            a[idx, :] = self.analog_signals[n].raw_data()

        return a

    def save(self, target_file):
        """
        Save the AnalogSignal (pickle or hickle, if available)
        :param target_file: filename
        """
        fp = tools.utils.data_handling.FileIO(target_file)
        fp.save(self)

    def mean(self, axis=0):
        """
        Return the mean AnalogSignal after having performed the average of all the signals
        present in the AnalogSignalList

        Examples:
            >> a.mean()

        See also:
            std

        :param axis: [0, 1], take the mean over time [0], or over neurons [1]
        """

        if axis == 0:
            result = np.zeros(int((self.t_stop - self.t_start) / self.dt), float)
            for id in self.id_list():
                result += self.analog_signals[id].signal
            return result/len(self)

        else:
            means = []
            for n in self.analog_signals:
                means.append(self.analog_signals[n].mean())

            return np.array(means)

    def std(self, axis=0):
        """
        Return the standard deviation along time between all the AnalogSignals contained in
        the AnalogSignalList

        Examples:
            >> a.std()
               numpy.array([0.01, 0.2404, ...., 0.234, 0.234]

        See also:
            mean
        """
        result = np.zeros((len(self), int(round((self.t_stop - self.t_start) / self.dt))), float)
        for count, id in enumerate(self.id_list()):
            try:
                result[count, :] = self.analog_signals[id].signal
            except ValueError:
                print("{0} {1}".format(result[count, :].shape, self.analog_signals[id].signal.shape))
                raise
        return np.std(result, axis)

    def event_triggered_average(self, eventdict, events_ids=None, analogsignal_ids=None, average=True,
                                t_min=0, t_max=100, mode='same'):
        """
        Returns the event triggered averages of the analog signals inside the list.
        The events can be a SpikeList object or a dict containing times.
        The average is performed on a time window t_spikes - tmin, t_spikes + tmax
        Can return either the averaged waveform (average = True), or an array of all the
        waveforms triggered by all the spikes.

        Inputs:
            events  - Can be a SpikeList object (and events will be the spikes) or just a dict
                      of times
            average - If True, return a single vector of the averaged waveform. If False,
                      return an array of all the waveforms.
            mode    - 'same': the average is only done on same ids --> return {'eventids':average};
                      'all': for all ids in the eventdict the average from all ananlog signals
                      is returned --> return {'eventids':{'analogsignal_ids':average}}
            t_min   - Time (>0) to average the signal before an event, in ms (default 0)
            t_max   - Time (>0) to average the signal after an event, in ms  (default 100)
            events_ids - when given only perform average over these ids
            analogsignal_ids = when given only perform average on these ids

        Examples
            >> vmlist.event_triggered_average(spikelist, average=False, t_min = 50, t_max = 150, mode = 'same')
            >> vmlist.event_triggered_average(spikelist, average=True, mode = 'all')
            >> vmlist.event_triggered_average({'1':[200,300,'3':[234,788]]}, average=False)
        """
        if isinstance(eventdict, SpikeList):
            eventdict = eventdict.spiketrains
        if events_ids is None:
            events_ids = list(eventdict.keys())
        if analogsignal_ids is None:
            analogsignal_ids = list(self.analog_signals.keys())

        x = np.ceil(np.sqrt(len(analogsignal_ids)))
        results = {}

        for id in events_ids:
            events = eventdict[id]
            if len(events) <= 0:
                continue
            if mode is 'same':
                if id in self.analog_signals and id in analogsignal_ids:
                    results[id] = self.analog_signals[id].event_triggered_average(events, average=average,
                                                                                  t_min=t_min, t_max=t_max)
            elif mode is 'all':
                results[id] = {}
                for id_analog in analogsignal_ids:
                    analog_signal = self.analog_signals[id_analog]
                    results[id][id_analog] = analog_signal.event_triggered_average(events, average=average,
                                                                                   t_min=t_min, t_max=t_max)
        return results

    def zero_pad(self, n_steps=10):
        """
        pad analog signals with zeros

        :return:
        """
        interval = n_steps * self.dt
        stim_array = tools.signals.pad_array(self.as_array(), add=int(n_steps))
        time_axis = np.arange(self.t_start, self.t_stop + interval, self.dt)
        sig = AnalogSignalList([], [], times=time_axis, dt=self.dt, t_start=min(time_axis),
                               t_stop=max(time_axis) + self.dt, dims=self.dimensions)
        for idx in range(stim_array.shape[0]):
            signal = AnalogSignal(stim_array[idx, :], dt=self.dt, t_start=self.t_start, t_stop=self.t_stop
                                                                                             + interval)
            sig.append(idx, signal)

        return sig

    # ###############################################################################
    def plot_random(self, idx=None, display=True, save=False):
        """
        Plot a single channel
        :return:
        """
        if idx is None:
            idx = np.random.permutation(self.id_list())[0]
        fig = pl.figure()
        t_axis = self.time_axis()
        s_data = self.analog_signals[int(idx)].raw_data()
        if len(t_axis) != len(s_data):
            t_axis = t_axis[:-1]
        pl.plot(t_axis, s_data)
        fig.suptitle('Channel {0}'.format(str(idx)))
        tools.visualization.helper.fig_output(fig, display, save)
        return idx

    def plot(self, as_array=False, fig=None, ax=None, display=True, save=False):
        """
        Plot the entire signal
        :return:
        """
        if fig is None:
            fig = pl.figure()
            fig.suptitle("AnalogSignalList")
        t_axis = self.time_axis()
        if not as_array:
            axes = [fig.add_subplot(self.dimensions, 1, nn+1) for nn in range(self.dimensions)]
            for idx in range(self.dimensions):
                s_data = self.analog_signals[int(idx)].raw_data()
                if len(t_axis) != len(s_data):
                    t_axis = t_axis[:-1]
                ax = axes[idx]
                ax.plot(t_axis, s_data)
                ax.set_ylabel('Amplitude')
                ax.set_xlim([min(t_axis), max(t_axis)])
                if idx != self.dimensions - 1:
                    ax.set_xticklabels([])
                    # ax.axis('off')
                else:
                    ax.set_xlabel('Time [ms]')
                    ax.set_ylabel('Amplitude')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.get_xaxis().tick_bottom()
                ax.get_yaxis().tick_left()
        else:
            if ax is None:
                ax = fig.add_subplot(111)
            tools.visualization.plotting.plot_matrix(self.as_array(), ax=ax, display=True)
        tools.visualization.helper.fig_output(fig, display, save)
