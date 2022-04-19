import numpy as np
import importlib

nest = None if importlib.util.find_spec("nest") is None else importlib.import_module("nest")
try:
    from nest import topology as tp
except:
    pass

from fna.tools import network_architect as net
from fna.tools import parameters


def setup_network_topology(population_dictionary, population_name, is_subpop=False):
    """
    Create a population with spatial topology (nest topology module), will be used when instantiating a Network object
    :param population_dictionary: [dict] population parameters dictionary
    :param population_name: [str] population label
    :param is_subpop: [bool] is this a sub-population?
    :return: Population object
    """
    tp_dict = population_dictionary['topology_dict']
    tp_dict.update({'elements': population_name})
    layer = tp.CreateLayer(tp_dict)
    gids = nest.GetLeaves(layer)[0]
    population_dictionary.update({'topology_dict': tp_dict, 'layer_gid': layer,
                        'is_subpop': is_subpop, 'gids': gids})
    return net.Population(parameters.ParameterSet(population_dictionary)), gids


def set_positions(N, dim=2, topology='random'):
    """
    Generates neuron's positions in space according to the desired topology
    :param N: total number of neurons
    :param dim: number of spatial dimensions
    :param topology: type of topology (random or lattice)
    :return:
    """
    if topology == 'random':
        # set neuron positions randomly in [0., sqrt(N)]
        pos = (np.sqrt(N) * np.random.random_sample((int(N), dim))).tolist()

    elif topology == 'lattice':
        # set neuron positions in a grid lattice (corresponding to integer points)
        assert (np.sqrt(N) % 1) == 0., 'Please choose a value of N with an integer sqrt..'
        xs = np.linspace(0., np.sqrt(N) - 1, int(np.sqrt(N)))
        if dim == 2:
            pos = [[x, y] for y in xs for x in xs]
        elif dim == 3:
            pos = [[x, y, z] for y in xs for x in xs for z in xs]
        else:
            raise NotImplementedError("Spatial arrangement can only assume 2 or 3 dimensions")
        np.random.shuffle(pos)
    else:
        raise NotImplementedError("{0!s} topology not implemented".format(topology))
    return pos


def get_center(network, layer_gid=None):
    """
    Get the global id of the central neuron
    :param network: network object
    :param layer_gid: global layer id (if None, all layers in network will be scanned)
    :return:
    """
    if layer_gid is None:
        layer_gid = [x.layer_gid[0] for x in network.populations]
    gid = tp.FindCenterElement(layer_gid)
    pos = tp.GetPosition(gid)
    return gid, pos


