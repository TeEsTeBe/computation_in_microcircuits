import io
import logging
import os
import pickle as pickle
import resource
import sys
from time import time
import psutil


def log_stats(cmd=None, flush_file=None, flush_path=None):
    """
    Dump some statistics (system, timers, etc) at the end of the run.

    :param cmd:
    :param flush_file:
    :param flush_path:
    :return:
    """
    logger.info('************************************************************')
    logger.info('************************ JOB STATS *************************')
    logger.info('************************************************************')

    if cmd:
        logger.info('Calling command: {}'.format(cmd))

    # system
    logger.info('')
    logger.info('System & Resources:')
    memory = {
        'psutil': None if 'psutil' not in sys.modules else memory_usage_psutil(),
        'peak-total-resource': memory_usage_resource(),
        'peak-self-resource': memory_usage_resource(with_children=False),
    }
    logger.info('\t\tMemory usage (MB, psutil): {}'.format(memory['psutil']))
    logger.info('\t\tPeak total memory usage (MB): {}'.format(memory['peak-total-resource']))
    logger.info('\t\tPeak (self) memory usage (MB): {}'.format(memory['peak-self-resource']))

    # timing
    logger.info('')
    logger.info('Timers:')
    global log_timer
    for name, timer in log_timer.get_all_timers().items():
        logger.info('\t\t{}: {} s'.format(name.capitalize(), timer['duration']))

    # flush to a main file
    if flush_file and flush_path:
        global main_log

        try:
            os.makedirs(flush_path)
        except OSError:
            if not os.path.isdir(flush_path):
                raise

        with open(os.path.join(flush_path, flush_file), 'w') as f:
            f.write(main_log.getvalue())

        with open(os.path.join(flush_path, flush_file.replace('.log', '_timers.pkl')), 'wb') as f:
            pickle.dump(log_timer, f)

        with open(os.path.join(flush_path, flush_file.replace('.log', '_memory.pkl')), 'wb') as f:
            pickle.dump(memory, f)


def update_log_handles(main=True, job_name=None, path=None):
    """
    Update log streams / files (including paths) according to how the program is run. If running from main.py,
    we buffer all output to `main_log` and will flush to file at the end. If this is a job (on a cluster), write
    directly to a file.
    :param main: if running from main.py, log to buffer and flush later
    :param job_name:
    :param path:
    :return:
    """
    if main:
        global main_log
        # logging.basicConfig(stream=main_log, level=logging.INFO)
        handler = logging.StreamHandler(main_log)
    else:
        handler = logging.FileHandler('{}/job_{}.log'.format(path, job_name))

    handler_format = logging.Formatter('[%(name)s - %(levelname)s] %(message)s')
    handler.setFormatter(handler_format)
    handler.setLevel(logging.INFO)

    for name, logger_ in logging.Logger.manager.loggerDict.items():
        # whether to log into the main file (always, unless it's a submitted job on a cluster
        if main:
            if isinstance(logger_, logging.Logger):
                logger_.addHandler(handler)
        else:
            logger_.addHandler(handler)
            logger_.propagate = False
            global run_local
            run_local = False


def get_logger(name):
    """
    Initialize a new logger called `name`.
    :param name: Logger name
    :return: logging.Logger object
    """
    logging.basicConfig(format='[%(filename)s:%(lineno)d - %(levelname)s] %(message)s'.format(name),
                        level=logging.INFO)
    logger_ = logging.getLogger(name)

    global main_log_handler
    logger_.addHandler(main_log_handler)

    return logger_


def memory_usage_psutil():
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem


def memory_usage_resource(with_children=True):
    self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.
    if with_children:
        return self + children
    else:
        return self


main_log = io.StringIO()
main_log_handler = logging.StreamHandler(main_log)

# initialize logging buffer as a stream handler
main_log_handler.setFormatter(logging.Formatter('[%(name)s:%(lineno)d - %(levelname)s] %(message)s'))
main_log_handler.setLevel(logging.INFO)


class Timer:
    """
    Time class
    """

    def __init__(self):
        self.timers = {}

    def start(self, name):
        """
        Starts a new timer at the current timepoint. If the timer already exists, the start time will be overwritten,
        otherwise a new timer entry is created.
        :param name:
        :return:
        """
        self.timers[name] = {
            'start': time(),
            'stop': 0.,
            'duration': 0.
        }

    def restart(self, name):
        """
        Resets the start time of an existing timer to the current timepoint and sets the stop time to None. If the timer
        doesn't yet exist, simply creates a new one by calling self.start().
        :param name:
        :return:
        """
        if name not in self.timers:
            self.start(name)
            return

        self.timers[name]['start'] = time()
        self.timers[name]['stop'] = None

    def stop(self, name):
        if name in self.timers:
            self.timers[name]['stop'] = time()
            self.timers[name]['duration'] = time() - self.timers[name]['start']

    def accumulate(self, name):
        if name in self.timers:
            self.timers[name]['stop'] = time()
            self.timers[name]['duration'] += time() - self.timers[name]['start']

    def get_all_timers(self):
        return self.timers


log_timer = Timer()
logger = get_logger(__name__)