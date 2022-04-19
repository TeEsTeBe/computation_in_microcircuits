import nest
import matplotlib.pyplot as plt


nest.Install('destexhemodule')

neuron_pars = {
    # 'V_T': -63.,
    # 'E_L': -80.,
    # 'C_m': 346.36,
    # 'g_L': 15.5862,
    # 'tau_syn_ex': 2.7,
    # 'tau_syn_in': 10.5,
    # # 't_ref': 2.0,
    # 'E_ex': 0.,
    # 'E_in': -75.,
    # 'E_Na': 50.,  # 35., 
    # 'g_Na': 17318.0,
    # 'E_K': -90.,
    # 'g_K': 3463.6,
    # 'g_M': 173.18,
    'g_M': 0.,
    # 'E_M': -80.,
    # 'g_noise_ex0': 12.,
    # 'g_noise_in0': 57.,
    # 'sigma_noise_ex': 3.,
    # 'sigma_noise_in': 6.6,
}

model_name = 'hh_cond_exp_destexhe'
neuron = nest.Create(model_name)
nest.SetStatus(neuron, neuron_pars)
multimeter = nest.Create('multimeter', 1, {'record_from': ['V_m']})
nest.Connect(multimeter, neuron)

nest.Simulate(2000.)

vm = nest.GetStatus(multimeter, 'events')[0]['V_m']

plt.plot(vm)
plt.title(model_name)
plt.xlabel('time [ms]')
plt.ylabel('Vm [mV]')
plt.savefig(f'{model_name}_vm_trace.pdf')

