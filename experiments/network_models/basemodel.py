from abc import ABC, abstractmethod

import nest


class BaseModel(ABC):
    """Abstract basis class which all models should inherit from to make sure all necessary functions are implemented"""

    base_neuron_pars = {
        'g_L': 15.5862,
    }

    iaf_neuron_parameters = {
        'C_m': 346.36,  # membrane capacity (pF)
        'E_L': -80.,  # resting membrane potential (mV)
        'I_e': 0.,  # external input current (pA)
        'V_m': -80.,  # membrane potential (mV)
        'V_reset': -80.,  # reset membrane potential after a spike (mV)
        'V_th': -55.,  # spike threshold (mV)
        't_ref': 3.0,  # refractory period (ms)
        'tau_syn_ex': 3.,  # membrane time constant (ms)
        'tau_syn_in': 6.,  # membrane time constant (ms)
        'g_L': 15.5862,
        'E_ex': 0.,
        'E_in': -75.,
    }

    # see "% create inputs" in make_network_V1_HH (matlab)
    # input(1).parameters.Synapse(EE).W == 3e-8
    # abs(E(nS)-V) = 65e-3
    # -> 4.6154e-7 -> 461.54e-9
    # input_weight = 461.54  # nS

    # input_weight = 1.9  # mV
    input_weight = 30. / base_neuron_pars['g_L']  # 1.924779612734342

    populations = None
    pop_counts = None
    neuron_pars = None
    conn_ids_from_to = None
    network_params_from_to = None

    def __init__(self):
        self.neuron_model = 'hh_cond_exp_destexhe'
        self.install_nest_modules()

    @abstractmethod
    def connect_input_stream1(self, spike_generators, scaling_factor):
        """ Function that defines how the first input stream is connected to the network model

        Parameters
        ----------
        spike_generators : NEST device list
            devices that generate the spikes of the first input stream`
        scaling_factor: float
            scaling factor for the connections from the first input stream to the network

        """

        pass

    @abstractmethod
    def connect_input_stream2(self, spike_generators, scaling_factor):
        """ Function that defines how the second input stream is connected to the network model

        Parameters
        ----------
        spike_generators : NEST device list
            devices that generate the spikes of the second input stream`
        scaling_factor: float
            scaling factor for the connections from the second input stream to the network

        """

        pass

    @abstractmethod
    def calculate_state_matrices(self):
        """ Compiles the state matrices and stores them. You need to call this function before get_state_matrices() """
        pass

    @abstractmethod
    def get_state_matrices(self, discard_steps, steps_per_trial):
        """ Returns the state matrices (you have to call calculate_state_matrices() first)

        Parameters
        ----------
        discard_steps: int
            number of warm up steps that should be discarded at the beginning
        steps_per_trial: int
            number of steps a single trial is made of

        Returns
        -------
        dict
            dictionary with state matrices

        """

        pass

    def install_nest_modules(self):
        """ Installs the neuron model if it is not already installed """
        if self.neuron_model not in nest.Models():
            nest.Install('destexhemodule')
            print('Destexhe NEST module installed')
        else:
            print('Destexhe NEST module has been installed before')
