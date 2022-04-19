import os
import pickle as pickle
import sys
import urllib

import numpy as np

from fna.tools.utils.logger import logger
from fna.tools.utils.operations import isiterable

data_path = None  # root path of the project's data folder
data_label = None  # global label of the experiment (root folder's name within the data path)
instance_label = None  # usually derived from the data_label, can contain extra parameters
filename_prefixes = None
paths = {}


def extract_data_fromfile(fname):
    """
    Extract raw data from a nest.device recording stored in file
    :param fname: filename or list of filenames
    """

    def get_data(filename, sepchar="\t", skipchar="#"):
        """
        Load data from a text file and returns a list of data
        """
        with open(filename, "r") as myfile:
            contents = myfile.readlines()
        data = []
        header = True
        idx = 0
        while header:
            if contents[idx][0] != skipchar:
                header = False
                break
            idx += 1
        for i in range(idx, len(contents)):
            line = contents[i].strip().split(sepchar)
            if i == 0:
                line_len = len(line)
            if len(line) == line_len:
                id = [float(line[0])]
                id += list(map(float, line[1:]))
                data.append(np.array(id))
        return np.array(data)

    if isiterable(fname):
        data = None
        for f in fname:
            print(("Reading data from file {0}".format(f)))
            if os.path.isfile(f) and os.path.getsize(f) > 0:
                with open(f, 'r') as fp:
                    if fp.readline(4) == '....':
                        info = fp.readlines()[1:]
                    else:
                        info = []
                if len(info):
                    with open(f, 'w') as fp:
                        fp.writelines(info)

                if data is None:
                    data = get_data(f)
                else:
                    data = np.concatenate((data, get_data(f)))
    else:
        with open(fname, 'r') as fp:
            if fp.readline(4) == '....':  # some bug in the recording...
                info = fp.readlines()[1:]
            else:
                info = []
        if len(info):
            with open(fname, 'w') as fp:
                fp.writelines(info)

        data = get_data(fname)
    return data


def remove_files(fname):
    """
    Remove all files in list
    :param fname:
    :return:
    """
    if isiterable(fname):
        for ff in fname:
            if os.path.isfile(ff) and os.path.getsize(ff) > 0:
                os.remove(ff)
    else:
        if os.path.isfile(fname) and os.path.getsize(fname) > 0:
            os.remove(fname)


def set_storage_locations(data_path_, data_label_, instance_label_=None, save=True):
    """
    Define paths to store data
    :param save: [bool] is False, no paths are created
    :return save_paths: dictionary containing all relevant storage locations
    """
    if save:
        logger.info("Setting storage paths...")
        main_folder = os.path.join(data_path_, data_label_)

        figures = main_folder + '/figures/'
        inputs = main_folder + '/inputs/'
        parameters = main_folder + '/parameters/'
        results = main_folder + '/results/'
        activity = main_folder + '/activity/'
        network = main_folder + '/system/'
        logs = main_folder + '/logs/'
        other = main_folder + '/other/'

        filename_prefixes_ = {
            'state_matrix': 'SM',
            'output_mapper': 'OM',
            'sequencer': 'SEQ',
            'embedding': 'EMB',
            'connectivity': 'CON'
        }

        dirs = {'main': main_folder, 'figures': figures, 'inputs': inputs, 'parameters': parameters,
                'results': results, 'activity': activity, 'logs': logs, 'other': other, 'system': network}

        for d in list(dirs.values()):
            try:
                os.makedirs(d)
            except OSError:
                pass

        dirs['label'] = data_label_

        global data_path
        global data_label
        global instance_label
        global filename_prefixes
        global paths

        data_path = data_path_
        data_label = data_label_
        if instance_label_ is not None:
            data_label = instance_label_
        instance_label = instance_label_
        filename_prefixes = filename_prefixes_
        paths = dirs

        return dirs
    else:
        logger.info("No data will be saved!")
        return {'label': False, 'figures': False, 'activity': False}


def download_dataset(file_urls, target_paths):
    """
    Download a dataset from the provided urls
    :param file_urls: [list of str] complete urls
    :param target_paths: [list of str] target storage locations
    :return:
    """
    logger.info("Downloading the dataset... (It may take some time)")

    for url, pth in zip(file_urls, target_paths):
        if not os.path.isfile(pth):
            logger.info("\t - Downloading {0}".format(url))
            urllib.request.urlretrieve(url, pth)

    logger.info("Done!")


class FileIO(object):
    """
    Standard file loading and saving. Handle hickle (h5py + pickle) or pickle dictionaries, depending what's
    available. This class simplifies data handling by providing a simple wrapper for pickle (hickle) save and load
    routines
    """

    def __init__(self, filename, compression=None):
        """
        Create the file object
        :param filename: full path to target file
        :param compression:
        """
        self.filename = filename
        self.compression = compression

    def __str__(self):
        return "%s" % self.filename

    def load(self):
        """
        Loads h5-file and extracts the dictionary within it.

        Outputs:
          dict - dictionary, one or several pairs of string and any type of variable,
                 e.g dict = {'name1': var1,'name2': var2}
        """
        print("Loading %s" % self.filename)
        with open(self.filename, 'r') as f:
            data = pickle.load(f)
            return data

    def save(self, data):
        """
        Stores a dictionary (dict), in a file (filename).
        Inputs:
          filename - a string, name of file to store the dictionary
          dict     - a dictionary, one or several pairs of string and any type of variable,
                     e.g. dict = {'name1': var1,'name2': var2}
        """
        with open(self.filename, 'wb') as f:
            pickle.dump(data, f)


def import_mod_file(full_path_to_module):
    """
    import a module from a path
    :param full_path_to_module:
    :return: imports module
    """
    module_dir, module_file = os.path.split(full_path_to_module)
    module_name, module_ext = os.path.splitext(module_file)
    sys.path.append(module_dir)
    try:
        module_obj = __import__(module_name)
        module_obj.__file__ = full_path_to_module
        return module_name, module_obj
    except Exception as er:
        raise ImportError("Unable to load module {0}, check if the name is repeated with other scripts in "
                          "path. Error is {1}".format(str(module_name), str(er)))
