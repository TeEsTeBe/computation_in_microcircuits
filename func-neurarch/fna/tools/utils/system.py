import numpy as np

from fna.tools import utils
from fna.tools.parameters import ParameterSet, extract_nestvalid_dict

logger = utils.logger.get_logger(__name__)

try:
    import nest
except ImportError as e:
    logger.warning("Could not import dependency: {}. Some functions may be unavailable!".format(e))


def set_kernel_defaults(resolution=0.1, run_type='local', data_label='', data_paths=None,
                        np_seed=None, **system_pars):
    """
    Return standardized kernel parameters dictionary according to the specifications
    :param resolution: simulator integration time step (float)
    :param run_type: system label (str)
    :param data_label: storage label (str)
    :param data_paths: system-specific storage locations (dict)
    :param np_seed: numpy seed, for reproducibility
    :return kernel_parameters: dict
    """
    if data_paths is None:
        logger.warning("No storage paths provided, all data will be stored in ./data/")
        data_paths = {'local': {
            'data_path': './data/',
            'jdf_template': None,
            'matplotlib_rc': None,
            'remote_directory': None,
            'queueing_system': None}}

    keys = ['nodes', 'ppn', 'queue']#, 'sim_time', 'transient_time']
    for k in keys:
        if k not in system_pars:
            raise TypeError("system parameters dictionary must contain the following keys {0}".format(str(keys)))

    N_vp = system_pars['nodes'] * system_pars['ppn']

    if not np_seed:
        np_seed = np.random.randint(1000000000) + 1
    np.random.seed(np_seed)
    msd = np.random.randint(100000000000)

    kernel_pars = {
        'resolution': resolution,
        'data_prefix': data_label,
        'data_path': data_paths[run_type]['data_path'],
        'mpl_path': data_paths[run_type]['matplotlib_rc'],
        'overwrite_files': True,
        'print_time': (run_type == 'local'),
        'rng_seeds': list(range(msd + N_vp + 1, msd + 2 * N_vp + 1)),
        'grng_seed': msd + N_vp,
        'total_num_virtual_procs': N_vp,
        'local_num_threads': system_pars['ppn'] if 'local_num_threads' not in system_pars
                                                else system_pars['local_num_threads'],
        'np_seed': np_seed,

        'system': {
            'local': (run_type == 'local'),
            'system_label': run_type,
            'queueing_system': data_paths[run_type]['queueing_system'],
            'jdf_template': data_paths[run_type]['jdf_template'],
            'remote_directory': data_paths[run_type]['remote_directory'],
            'jdf_fields': {}
        }}

    for k, v in system_pars.items():
        kernel_pars['system']['jdf_fields'].update({'{{ '+'{0}'.format(k)+' }}': str(v)})

    return ParameterSet(kernel_pars)


def reset_nest_kernel(kernel_pars=None):
    """
    Reset the NEST kernel
    :params kernel_pars: kernel parameters dictionary or Nonec
    :return:
    """
    nest.ResetKernel()
    nest.set_verbosity('M_WARNING')
    if kernel_pars is not None:
        nest.SetKernelStatus(extract_nestvalid_dict(kernel_pars.as_dict(), param_type='kernel'))


def update_generator_states(model, gids, signal_amplitudes, signal_times):
    """
    Update the state of a NEST generator device
    :param model: generator model name
    :param gids: gids of generators to update
    :param signal_amplitudes:
    :param signal_times:
    :return:
    """
    if model == 'step_current_generator':
        nest.SetStatus(gids, {'amplitude_times': signal_times, 'amplitude_values': signal_amplitudes})
    elif model == 'inhomogeneous_poisson_generator':
        nest.SetStatus(gids, {'rate_times': signal_times, 'rate_values': signal_amplitudes})