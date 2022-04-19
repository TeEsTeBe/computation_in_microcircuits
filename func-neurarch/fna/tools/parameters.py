"""
========================================================================================================================
Parameters Module
========================================================================================================================
(incomplete documentation)

Module for dealing with model parameters

Classes
-------
Parameter 		 - simple, single parameter class
ParameterRange 	 - specify list or array of possible values for a given parameter
ParameterSet 	 - represent/manage hierarchical parameter sets
ParameterDerived - specify a parameter derived from the value of another

Functions
---------
set_params_dict - import multiple parameter dictionaries from a python script and gathers them all in	a single
dictionary, which can then be used to create a ParameterSet object
extract_nestvalid_dict - verify whether the parameters dictionary are in the correct format, with adequate keys,
in agreement with the nest parameters dictionaries so that they can later be passed as direct input to nest
import_mod_file - import a module given the path to its file
isiterable - check whether input is iterable (list, dictionary, array...)
nesteddict_walk - walk a nested dictionary structure and return an iterator
contains_instance - check whether instances are part of object
nesteddict_flatten - flatten a nested dictionary
load_parameters - easiest way to create a ParameterSet from a dictionary file
string_table - convert a table written as multi-line string into a nested dictionary
copy_dict - copies a dictionary and updates it with extra key-value pairs
========================================================================================================================
"""
import collections
import itertools
import os
import types
import numpy as np
import importlib

nest = None if importlib.util.find_spec("nest") is None else importlib.import_module("nest")

try:
    import ntpath
except ImportError as e:
    raise ImportError("Import dependency not met: {}".format(e))

from fna.tools.utils import data_handling, operations, logger

# np.set_printoptions(threshold=np.nan, suppress=True)
logger = logger.get_logger(__name__)


##########################################################################################
def validate_keys(dictionary, required_keys=None, any_keys=None):
    """
    Raise an error if at least one key in `keys` is not found in `dictionary`.
    :param required_keys:
    :param any_keys:
    :param dictionary: [dictionary or ParameterSet object]
    :return:
    """
    for k in required_keys:
        if k not in dictionary:
            raise KeyError('Required parameter {} not found in parameter set {}'.format(k, str(dictionary)))

    if any_keys is None:
        return

    for k in any_keys:
        if k in dictionary:
            return

    raise KeyError('None of possible parameters {} not found in parameter set {}'.format(any_keys, str(dictionary)))


def set_params_dict(source_file):
    """
    Import multiple parameter dictionaries from a python script and gathers them all in
    a single dictionary, which can then be used to create a ParameterSet object.

    :param source_file: [string] path+filename of source script or, if source script is in current directory,
                        it can be given as just the filename without extension
    :return: full dictionary
    """
    d = dict()

    if os.path.isfile(source_file):
        module_name, _ = data_handling.import_mod_file(source_file)
    else:
        __import__(source_file)
        module_name = source_file

    for attr, val in eval(module_name).__dict__.items():
        if isinstance(val, dict) and attr != '__builtins__':
            d[str(attr)] = val

    return d


def extract_nestvalid_dict(d, param_type='neuron'):
    """
    Verify whether the parameters dictionary are in the correct format, with adequate keys, in agreement with the nest
    parameters dictionaries so that they can later be passed as direct input to nest.

    :param d: parameter dictionary
    :param param_type: type of parameters - kernel, neuron, population, network_architect, connections, topology
    :return: valid dictionary
    """
    if param_type == 'neuron' or param_type == 'synapse' or param_type == 'device':
        assert d['model'] in nest.Models(), "Model %s not currently implemented in %s" % (d['model'], nest.version())

        accepted_keys = list(nest.GetDefaults(d['model']).keys())
        accepted_keys.remove('model')
        nest_dict = {k: v for k, v in d.items() if k in accepted_keys}
    elif param_type == 'kernel':
        accepted_keys = list(nest.GetKernelStatus().keys())
        nest_dict = {k: v for k, v in d.items() if k in accepted_keys}
    else:
        # TODO
        logger.error("{!s} not implemented yet".format(param_type))
        exit(-1)

    return nest_dict


def load_parameters(parameters_url, **modifications):
    """
    Load a ParameterSet from a url to a text file with parameter dictionaries.

    :param parameters_url: [str] full path to file
    :param modifications: [path=value], where path is a . (dot) delimited path to
    a parameter in the  parameter tree rooted in this ParameterSet instance
    :return: ParameterSet
    """
    parameters = ParameterSet(parameters_url)
    parameters.replace_values(**modifications)

    return parameters


def remove_all_labels(parameter_set):
    """
    Removes all the 'label' entries from the parameter set.

    :param parameter_set:
    :return:
    """
    new_pars = {k: v for k, v in list(parameter_set.items()) if k != 'label'}
    for k, v in list(parameter_set.items()):
        if k != 'label':
            if isinstance(v, dict) and 'label' in list(v.keys()):
                # for k1, v1 in v.items():
                # if k1 != 'label':
                new_pars[k] = {k1: v1 for k1, v1 in list(v.items()) if k1 != 'label'}
            new_pars[k] = v
    return ParameterSet(new_pars)


######################################################################################
class Parameter(object):
    """
    Simpler class specifying single parameters
    """

    def __init__(self, name, value, units=None):
        self.name = name
        self.value = value
        self.units = units
        self.type = type(value)

    def __repr__(self):
        s = "%s = %s" % (self.name, self.value)
        if self.units is not None:
            s += " %s" % self.units
        return s


#########################################################################################
class ParameterSet(dict):
    """
    Class to manage complex parameter sets.

    Usage example:

        > sim_params = ParameterSet({'dt': 0.1, 'tstop': 1000.0})
        > exc_cell_params = ParameterSet({'tau_m': 20.0, 'cm': 0.5})
        > inh_cell_params = ParameterSet({'tau_m': 15.0, 'cm': 0.5})
        > network_params = ParameterSet({'excitatory_cells': exc_cell_params, 'inhibitory_cells': inh_cell_params})
        > P = ParameterSet({'sim': sim_params, 'network': network_params})
        > P.sim.dt
        0.1
        > P.network.inhibitory_cells.tau_m
        15.0
    """

    @staticmethod
    def read_from_file(pth, filename):
        """
        Import parameter dictionary stored as a text file. The
        file must be in standard format, i.e. {'key1': value1, ...}
        :param pth: path
        :param filename:
        :return:
        """

        assert os.path.exists(pth), "Incorrect path..."

        with open('{}/{}'.format(pth, filename), 'r') as fi:
            contents = fi.read()
        d = eval(contents)

        return d

    def __init__(self, initializer, label=None):
        """

        :param initializer: parameters dictionary, or string locating the full path of text file with
        parameters dictionary written in standard format
        :param label: label the parameters set
        :return: ParameterSet
        """

        super().__init__()

        def convert_dict(d, label):
            """
            Iterate through the dictionary d, replacing all items by ParameterSet objects
            :param d:
            :param label:
            :return:
            """
            for k, v in list(d.items()):
                if isinstance(v, ParameterSet):
                    d[k] = v
                elif isinstance(v, dict):
                    d[k] = convert_dict(d, k)
                else:
                    d[k] = v
            return ParameterSet(d, label)

        # self._url = None
        if isinstance(initializer, str):  # url or str
            if os.path.exists(initializer):
                pth = os.path.dirname(initializer)
                if pth == '':
                    pth = os.path.abspath('')
                filename = ntpath.basename(initializer)
                initializer = ParameterSet.read_from_file(pth, filename)

        # initializer is now a dictionary (if it was a path before), and we can iterate it and replace all items by
        # ParameterSets
        if isinstance(initializer, dict):
            for k, v in list(initializer.items()):
                if isinstance(v, ParameterSet):
                    self[k] = v

                elif isinstance(v, dict):
                    self[k] = ParameterSet(v)

                else:
                    self[k] = v
        else:
            raise TypeError("initializer must be either a string specifying "
                            "the full path of the parameters file, "
                            "or a parameters dictionary")

        # set label
        if isinstance(initializer, dict):
            if 'label' in initializer:
                self.label = initializer['label']
            # else:
            #     self.label = label
        elif hasattr(initializer, 'label'):
            self.label = label or initializer.label
        # else:
        #     self.label = label

    def flat(self):
        __doc__ = operations.nesteddict_walk.__doc__
        return operations.nesteddict_walk(self)

    def flatten(self):
        __doc__ = operations.nesteddict_flatten.__doc__
        return operations.nesteddict_flatten(self)

    def __eq__(self, other):
        """
        Simple function for equality check of ParameterSet objects. Check is done implicitly by converting the objects
        to dictionaries.
        :param other:
        :return:
        """
        # compare instances
        if not isinstance(self, other.__class__):
            return False

        ad, od = self.as_dict(), other.as_dict()
        for key in ad:
            if ad[key] != od[key]:
                return False
        return True

    def __getattr__(self, name):
        """
        Allow accessing parameters using dot notation.
        """
        try:
            return self[name]
        except KeyError:
            return self.__getattribute__(name)

    def __setattr__(self, name, value):
        """
        Allow setting parameters using dot notation.
        """
        self[name] = value

    def __getitem__(self, name):
        """
        Modified get that detects dots '.' in the names and goes down the
        nested tree to find it
        """
        # TODO this?
        if isinstance(name, tuple):
            return dict.__getitem__(self, name)

        split = name.split('.', 1)
        if len(split) == 1:
            return dict.__getitem__(self, name)
        # nested get
        return dict.__getitem__(self, split[0])[split[1]]

    def flat_add(self, name, value):
        """
        Like `__setitem__`, but it will add `ParameterSet({})` objects
        into the namespace tree if needed.
        """

        split = name.split('.', 1)
        if len(split) == 1:
            dict.__setitem__(self, name, value)
        else:
            # nested set
            try:
                ps = dict.__getitem__(self, split[0])
            except KeyError:
                # setting nested name without parent existing
                # create parent
                ps = ParameterSet({})
                dict.__setitem__(self, split[0], ps)
            # and try again
            ps.flat_add(split[1], value)

    def __setitem__(self, name, value):
        """
        Modified set that detects dots '.' in the names and goes down the
        nested tree to set it
        """
        if isinstance(name, str):
            split = name.split('.', 1)
            if len(split) == 1:
                dict.__setitem__(self, name, value)
            else:
                # nested set
                dict.__getitem__(self, split[0])[split[1]] = value
        # TODO maybe remove if unnecessary?
        elif isinstance(name, tuple):
            dict.__setitem__(self, name, value)

    def update(self, E, **F):
        """
        Update ParameterSet with dictionary entries
        """
        if isinstance(E, collections.abc.Mapping):
            for k in E:
                self[k] = E[k]
        else:
            for (k, v) in E:
                self[k] = v
        for k in F:
            self[k] = F[k]

    def __getstate__(self):
        """
        For pickling.
        """
        return self

    def save(self, url=None):
        """
        Write the ParameterSet to a text file
        The text format should be valid python code...
        :param url: locator (full path+ filename, str) - if None, data will be saved to self._url
        """

        if not url:
            print("Please provide url")
        assert url != ''

        with open(url, 'w') as fp:
            fp.write(self.pretty())

    def pretty(self, indent='   '):
        """
        Return a unicode string representing the structure of the `ParameterSet`.
        evaluating the string should recreate the object.
        :param indent: indentation type
        :return: string
        """

        def walk(d, indent, ind_incr):
            s = []
            for k, v in list(d.items()):
                if isinstance(v, list):
                    if k == 'randomize' and np.mean([isinstance(x, dict) for x in v]) == 1.:
                        s.append('%s"%s": [' % (indent, k))
                        for x in v:
                            s.append('{')
                            s.append(walk(x, indent + ind_incr, ind_incr))
                            s.append('},\n')
                        s.append('],\n')
                        v_arr = np.array([])
                        continue
                    elif np.mean([isinstance(x, list) for x in v]):
                        v_arr = np.array(list(itertools.chain(*v)))
                    else:
                        v_arr = np.array(v)
                    mark = np.mean([isinstance(x, tuple) for x in v])
                    if (v_arr.dtype == object) and mark:
                        if len(v) == 1:
                            if v and isinstance(v[0], tuple) and isinstance(v[0][0], types.BuiltinFunctionType):
                                if isinstance(v_arr.any(), types.BuiltinFunctionType):
                                    if v_arr.any().__name__ in np.random.__all__:
                                        s.append('%s"%s": [(%s, %s)],' % (indent, k, 'np.random.{0}'.format(v_arr.any(
                                        ).__name__), str(v_arr.all())))
                                elif isinstance(v_arr.any(), types.MethodType) and v_arr.any().__str__().find('scipy'):
                                    s.append('%s"%s": [(%s, %s)],' % (indent, k, 'st.{0}.rvs'.format(str(v_arr.any().__self__.name)),
                                                                      str(v_arr.all())))
                        else:
                            if v and isinstance(v_arr.any(), tuple):
                                if isinstance(v_arr.any()[0], types.BuiltinFunctionType):
                                    list_idx = np.where(v_arr.any() in v)[0][0]
                                    string = '%s"%s": [' % (indent, k)
                                    for idx, nnn in enumerate(v):
                                        if idx == list_idx:
                                            tmp = np.array(nnn)
                                            string += '(%s, %s), ' % ('np.random.{0}'.format(tmp.any().__name__), str(tmp.all()))
                                        elif nnn is not None and isinstance(nnn[0], types.BuiltinFunctionType):
                                            tmp = np.array(nnn)
                                            string += '(%s, %s), ' % ('np.random.{0}'.format(tmp.any().__name__), str(tmp.all()))
                                        else:
                                            string += '%s, ' % (str(nnn))
                                    string += '], '
                                    s.append(string)
                            elif v and isinstance(v_arr.any(), types.BuiltinFunctionType) and isinstance(v_arr.all(),
                                                                                                         dict):
                                string = '%s"%s": [' % (indent, k)
                                for idx, nnn in enumerate(v):
                                    tmp = np.array(nnn)
                                    string += '(%s, %s), ' % (
                                        'np.random.{0}'.format(tmp.any().__name__), str(tmp.all()))
                                string += '], '
                                s.append(string)
                    else:
                        if np.mean([isinstance(x, np.ndarray) for x in v]):
                            string = '%s"%s": [' % (indent, k)
                            for idx, n in enumerate(v):
                                if isinstance(n, np.ndarray):
                                    string += 'np.' + repr(n) + ', '
                                else:
                                    string += str(n) + ', '
                            string += '], '
                            s.append(string)
                        elif hasattr(v, 'items'):
                            s.append('%s"%s": {' % (indent, k))
                            s.append(walk(v, indent + ind_incr, ind_incr))
                            s.append('%s},' % indent)
                        elif isinstance(v, str):
                            s.append('%s"%s": "%s",' % (indent, k, v))
                        else:
                            # what if we have a dict or ParameterSet inside a list? currently they are not expanded.
                            # Should they be?
                            s.append('%s"%s": %s,' % (indent, k, v))

                elif isinstance(v, types.BuiltinFunctionType):
                    if v.__name__ in np.random.__all__:
                        s.append('%s"%s"' % (indent, 'np.random.{0}'.format(v.__name__)))
                else:
                    if hasattr(v, 'items'):
                        s.append('%s"%s": {' % (indent, k))
                        s.append(walk(v, indent + ind_incr, ind_incr))
                        s.append('%s},' % indent)
                    elif isinstance(v, str):
                        s.append('%s"%s": "%s",' % (indent, k, v))
                    elif isinstance(v, np.ndarray):
                        s.append('%s"%s": %s,' % (indent, k, repr(v)[6:-1]))
                    elif isinstance(v, tuple) and (isinstance(v[0], types.MethodType)
                                                   or isinstance(v[0], types.BuiltinFunctionType)):
                        v_arr = np.array(v)
                        if isinstance(v[0], types.MethodType):
                            s.append('%s"%s": (%s, %s),' % (indent, k, 'st.{0}.rvs'.format(
                                str(v_arr.any().__self__.name)), str(v_arr.all())))
                        elif isinstance(v[0], types.BuiltinFunctionType):
                            s.append('%s"%s": (%s, %s),' % (indent, k, 'np.random.{0}'.format(v_arr.any().__name__),
                                                            str(v_arr.all())))
                    else:
                        # what if we have a dict or ParameterSet inside a list? currently they are not expanded.
                        #  Should they be?
                        s.append('%s"%s": %s,' % (indent, k, v))
            return '\n'.join(s)

        return '{\n' + walk(self, indent, indent) + '\n}'

    def tree_copy(self):
        """
        Return a copy of the `ParameterSet` tree structure.
        Nodes are not copied, but re-referenced.
        """

        tmp = ParameterSet({})
        for key in self:
            value = self[key]
            if isinstance(value, ParameterSet):
                tmp[key] = value.tree_copy()
            else:
                tmp[key] = value
        return tmp

    def as_dict(self):
        """
        Return a copy of the `ParameterSet` tree structure as a nested dictionary.
        """
        tmp = {}

        for key in self:
            if not isinstance(self[key], types.BuiltinFunctionType):
                value = self[key]
                if isinstance(value, ParameterSet):
                    tmp[key] = value.as_dict()
                else:
                    tmp[key] = value
        return tmp

    def replace_values(self, **args):
        """
        This expects its arguments to be in the form path=value, where path is a
        . (dot) delimited path to a parameter in the  parameter tree rooted in
        this ParameterSet instance.

        This function replaces the values of each parameter in the args with the
        corresponding values supplied in the arguments.
        """
        for k in list(args.keys()):
            self[k] = args[k]

    def clean(self, termination='pars'):
        """
        Remove fields from ParameterSet that do not contain the termination
        This is mostly useful if, in the specification of the parameters
        additional auxiliary variables were set, and have no relevance for
        the experiment at hand...
        """
        accepted_keys = [x for x in self.keys() if x[-len(termination):] == termination]
        new_dict = {k: v for k, v in self.items() if k in accepted_keys}

        return ParameterSet(new_dict)

