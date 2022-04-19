# Characteristic columnar connectivity caters to cortical computation

This repository contains the code that you need to run the experiments presented in the paper *Characteristic columnar connectivity caters to cortical computation: replication, simulation and evaluation of a microcircuit model* by
Tobias Schulte to Brinke, Renato Duarte and Abigail Morrison.


To run this code, you have to install [NEST 3](https://nest-simulator.readthedocs.io/en/v3.0/installation/index.html) and use [nestml](https://nestml.readthedocs.io/en/v4.0/) to install the 
`hh_cond_exp_destexhe` neuron model first. You can find it in `nestml-models/destexhemodels`. Besides that you have to install the modified version of the [Functional Neural Architectures](https://zenodo.org/record/5752597) library
by going into the `func-neurarch` folder and calling `pip install .`.

You also have to install the other requirements from the `requirements.txt` file.

Then add this repository to your PYTHONPATH:
```commandline
export PYTHONPATH=$PYTHONPATH:/path/to/this/repository
```

## Running simulations
The main entry point for the simulations is the `run_tasks.py` script in the `experiments` folder.
To define which network you want to run you have to set the `--network_name` parameter. 
Possible values are `microcircuit`, `amorphous`, `degreecontrolled`, `degreecontrolled_no_io_specificity`, `smallworld`, 
`microcircuit_static` and `microcircuit_random_dynamics`.
With the `--group_name` parameter you can group multiple runs into a group to store them into the same parent folder.
This makes it easier to create plots for different experiments. 
The `--runtitle` parameter defines a prefix for the results folder of this specific run.

### Spike pattern tasks
For example to run the spike pattern task with the amorphous circuit and default parameters you can run the following command:

```commandline
python run_tasks.py --network_name amorphous --group_name mygroup --runtitle myrun1
```
This would store the results in `experiments/data/task_results/mygroup/myrun1_<parameters>`

### Firing rate tasks
If you want to run the firing rate based task instead of the spike pattern tasks you have to add the `--rate_tasks` parameter.

### Different neuron models
To disable the inctrinsic noise mechanism of the Hodgkin-Huxley neurons, you have to add the `--disable_conductance_noise` 
parameter and if you want to use the integrate-and-fire neurons you have to add `--neuron_model iaf_cond_exp`.
Also change the `--group_name` for the different neuron models to make sure that their results are not mixed in later steps.

### Detailed memory tasks
The detailed memory tasks use a shorter step duration of 5 ms and the task is to classify the patterns up to a delay of 14 steps.
This can be done by adding `--step_duration 5` and `--max_delay 14` to the call for the spike pattern classification task.
```commandline
python run_tasks.py --group_name memorytasks --network_name microcircuit --group_name mygroup --runtitle myrun1 --step_duration 5 --max_delay 14
```

### Other parameters
To get more information about the parameters you can run
```commandline
python run_tasks.py --help
```
result:
```
usage: run_tasks.py [-h] [--network_name NETWORK_NAME] [--N N] [--S_rw S_RW] [--S1 S1] [--S2 S2] [--reset_neurons] [--reset_synapses] [--steps_per_trial STEPS_PER_TRIAL] [--discard_steps DISCARD_STEPS]
                    [--train_trials TRAIN_TRIALS] [--test_trials TEST_TRIALS] [--step_duration STEP_DURATION] [--raster_plot_duration RASTER_PLOT_DURATION] [--max_delay MAX_DELAY]
                    [--spike_statistics_duration SPIKE_STATISTICS_DURATION] [--runtitle RUNTITLE] [--num_threads NUM_THREADS] [--gM_exc GM_EXC] [--gM_inh GM_INH] [--ok_if_folder_exists] [--rate_tasks]
                    [--input_dimensions INPUT_DIMENSIONS] [--freeze_last_input] [--start_s2 START_S2] [--group_name GROUP_NAME] [--neuron_model NEURON_MODEL] [--disable_conductance_noise]

optional arguments:
  -h, --help            show this help message and exit
  --network_name NETWORK_NAME
                        Network model, which should solve the task. Possible values: microcircuit, amorphous, degreecontrolled, degreecontrolled_no_io_specificity, smallworld, smallworld_norandomweight,microcircuit_static,
                        microcircuit_random_dynamics
  --N N                 Number of neurons in the network. Default: 560
  --S_rw S_RW           Scaling factor for the recurrent weights. Default: 66825/N (static synapses: 66825/N/73)
  --S1 S1               Scaling factor for the weights of the connections from the first input stream
  --S2 S2               Scaling factor for the weights of the connections from the second input stream
  --reset_neurons       If set the neuron states are reset after every trial
  --reset_synapses      If set the synapse states are reset after every trial
  --steps_per_trial STEPS_PER_TRIAL
                        Defines how many steps there are per trial. Default: 15
  --discard_steps DISCARD_STEPS
                        Defines how many steps are discarded at the beginning.
  --train_trials TRAIN_TRIALS
                        Number of trials used to train the readouts. Default: 1500
  --test_trials TEST_TRIALS
                        Number of trials used to test the readouts. Default: 300
  --step_duration STEP_DURATION
                        Duration of a single step in milliseconds. Default: 30.
  --raster_plot_duration RASTER_PLOT_DURATION
                        Defines for how long (in milliseconds) spikes are plottet. Default: 450.
  --max_delay MAX_DELAY
                        Maximum delay (in steps) for memory tasks
  --spike_statistics_duration SPIKE_STATISTICS_DURATION
                        Defines for how long (in milliseconds) spikes are detectedand used for spike statistic calculations. Default: 10000.
  --runtitle RUNTITLE   Title/name of the run. This is added to the results folder name
  --num_threads NUM_THREADS
                        Number of threads used by NEST.
  --gM_exc GM_EXC       Peak conductance of Mainen potassium ion channel for excitatory neurons. Default: 100.
  --gM_inh GM_INH       Peak conductance of Mainen potassium ion channel for inhibitory neurons. Default: 0 (channel deactivated)
  --ok_if_folder_exists
                        If set and the results folder already exists, previous results will be overwritten (mainly for testing)
  --rate_tasks          If set, the firing rate tasks are processed instead of the spike pattern classification tasks.
  --input_dimensions INPUT_DIMENSIONS
                        Number of spike trains per input stream. Default for spike pattern classification is 40 and for rate tasks is 4
  --freeze_last_input   If set, the spike patterns of the last step for each trial is fixed
  --start_s2 START_S2   Start time of input stream 2 in ms. Default value is 0 and should not be changed,because this messes up the task results. This is only needed to get a rasterplot like in the Häusler Maass 2006 paper.
  --group_name GROUP_NAME
                        Name of the group of experiments. This is also the parent folder name inside the data folder, where the results are stored.
  --neuron_model NEURON_MODEL
                        Neuron model (hh_cond_exp_destexhe or iaf_cond_exp)
  --disable_conductance_noise
                        Disables the conductance noise in the HH neuron
```

## Creating job scripts
You don't have to create job scripts for every simulation you want to do by hand. You can adjust the `jobscript_template.sh` file in the `experiments/scripts` folder and
run the `create_jobscripts.sh` script. This will generate the job scripts for all simulations which are needed to get the
results presented in the paper. There are two placeholder in the jobscript template, which will be replaced by `create_jobscripts.sh`
for every run. `%%%NAME%%%` will be replaced by the name of the corresponding run. `%%%PARAMS%%%` will be replaced by the parameters that
will be handed over to the `run_tasks.py` script. You also have to fill in the path to the repository and the path to your NEST
installation. Inside the `create_jobscripts.sh` file you can also change the values for `idstart` and `idend`. 
These parameters define the ids of the first and last run for every parameter set. For example `idstart=1` and `idend=10` will
create 10 runs for each parameter set and number the runs with all values between 1 and 10.


## Result tables (Tables 6 and 7)
When all runs from above are finished you can compile the results in tables by calling the `calculate_results_for_all_networks.py`
script with the corresponding group name parameter.
```commandline
python calculate_results_for_all_networks.py mygroup
```
This will result in the four CSV tables `<GROUPNAME>_task_differences.csv`, `<GROUPNAME>_task_differences_percent_rounded.csv` 
(content of table 7),`<GROUPNAME>average_task_results.csv` and `<GROUPNAME>_average_task_results_rounded.csv` (content of table 6).
These tables are stored in the `experiments/data` folder. They contain the averaged task results for all tasks and the difference
in the results to the data-based microcircuit model.

## Creating figures

### In- and outdegrees (Figure 2)
To create the plot with the histograms of degrees for each network you have to run the `plot_in_and_outdegrees.py` script inside the `experiments` folder.
You can adjust the number of runs for each network to average over by setting the `--n_trials` parameter. For the figure in 
the paper we used 100 runs.
If you have done a previous run and want to reuse the generated data you can add the path to this data with `--previous_data_path`.
```python
python plot_in_and_outdegrees.py --n_trials 100
```


### Raster plot and firing rate histograms (Figures 4 and 7)
Spike detector events are stored and Raster plots and firing rate histograms are automatically plotted for every run. You can find them in the results folder.
If you are only interested in these plots you should reduce the number of training and testing steps. 
By setting the `--start_s2` parameter to `100.` the second input stream starts 100 ms later as it is reported in the original paper.
```python
python run_tasks.py --train_trials 1 --test_trials 1 --start_s2 100.
```
Since the version which is plotted with every run is sligthly different from the ones in the paper you have to run the `plot_raster_and_firing_rates.py`
script and change the paths to the spike detector pickle files. The plots in Figure 4 are a bit bigger than in Figure 8 and 9 and to 
get the bigger ones you have to add `large` as a parameter to the scirpt:
```commandline
python plot_raster_and_firing_rates.py large
```
The figures are stored in `experiments/data/figures/raster_and_firing`


### Bar plot for main results with microcircuit and amorphous circuit (Figure 5)
Before you can create this plot you have to run the corresponding experiments.
If you want to average over multiple trials, run the following command multiple times and change the runtitle every time (for example run1, run2, ...).

- spike tasks microcicuit:
```commandline
python run_tasks.py --group_name hhneuron --network_name microcircuit --runtitle run1
```
- spike tasks amorphous circuit:
```commandline
python run_tasks.py --group_name hhneuron --network_name amorphous --runtitle run1
```
- rate tasks microcircuit:
```commandline
python run_tasks.py --group_name hhneuron --network_name microcircuit --rate_tasks --runtitle run1
```
- rate tasks amorphous circuit:
```commandline
python run_tasks.py --group_name hhneuron --network_name amorphous --rate_tasks --runtitle run1
```

You also need the results for the two other neuron models. As already mentioned in the Running Simulations section, 
you can do this by adding the `--disable_conductance_noise` parameter or adding `--neuron_model iaf_cond_exp`.
In addition, change the first part of the `--group_name` parameter accordingly. For example:
```commandline
python run_tasks.py --group_name iafneuron --network_name amorphous --neuron_model iaf_cond_exp --runtitle run1
```
or 
```commandline
python run_tasks.py --group_name nonoiseneuron --network_name microcircuit --disable_conductance_noise --runtitle run1
```

After running all of these experiments, you have to run the `plot_task_results.py` script and give the neuron specific
groupnames as values for the parameters `--hh_group`, `--nonoise_group` and `--iaf_group`:
```commandline
python plot_task_results_diff_neurons.py --hh_group hhneuron --nonoise_group nonoiseneuron --iaf_group iafneuron
```
The resulting figure is stored under `experiments/data/figures/diffneurons_task_results_mc_am/diffneurons_taskresults_mc_amorphous.pdf`.

### Task performance for different training trials (Figure 6A)
To run the necessary experiments you can use the following commands.
Like before you can run multiple trials per configuration if you vary the runtitle.
In the paper <NUM_TRAIN> is set to values between 40 and 480 in steps of 40.

- Microcircuit
```commandline
python run_tasks.py --group_name different_steps --network_name microcircuit --train_trials <NUM_TRAIN> --steps_per_trial 2 --step_duration 100. --input_dimension 4 --freeze_last_input --runtitle run1
```
- Amorphous circuit
```commandline
python run_tasks.py --group_name different_steps --network_name amorphous --train_trials <NUM_TRAIN> --steps_per_trial 2 --step_duration 100. --input_dimension 4 --freeze_last_input --runtitle run1
```

When you have the results of the experiments you can run the `plot_task_results_different_train_steps.py` script with the groupname as parameter:
```commandline
python plot_task_results_different_train_steps.py different_steps
```
The figure is stored to `experiments/data/figures/different_training_steps/different_training_steps.pdf`

### XOR performance for different network sizes (Figure 6B)
Also for this plot you have to run the coresponding experiments first.
Set <NETWORK_SIZE> one by one to every value of [160, 360, 560, 810, 1000, 2000, 5000, 10000]. 
For each network size you can run multiple trials by changing the `--runtitle` to different values (for example run1, run2, ...).

- Microcircuit
```commandline
python run_tasks.py --group_name different_N --network_name microcircuit --N <NETWORK_SIZE> --runtitle run1
```
- Amorphous circuit
```commandline
python run_tasks.py --group_name different_N --network_name amorphous --N <NETWORK_SIZE> --runtitle run1
```

After running these experiments you can run the `plot_task_results_different_N.py` script with the groupname as a parameter
to get the plot.
```commandline
python plot_task_results_different_N.py different_N
```
If you have changed the group name in the above experiments you have to change the `mc_group` and `am_group` variables inside `plot_task_resutls_different_N`.
The figure is then stored at `experiments/data/figures/XOR_performance_different_N/xor_performance_different_N.pdf`.

### Task performances for detailed memory task (Figure 8)
To get line plots for all the networks for the detailed memory tasks like in Figure 10A, you have to run the 
`plot_memory_task_results.py` script with the group name as parameter:
```commandline
python plot_memory_task_results.py memorytasks
```
This creates the figure `memory_task_lines.pdf` in the folder `experiments/data/figures/memory_tasks`.
In the same folder a `results_full.pkl` file is created, which is needed for the next figures.
After doing this, we run the `plot_memory_summary.py` script and give the path to the `results_full.pkl` file as the 
first parameter and the path to the `<GROUPNAME>_average_task_results.csv` file, which we created in the 'Result tables (Tables 6 and 7)'
section, as a second parameter.
```commandline
python plot_memory_summary.py data/figures/memory_tasks/results_full.pkl data/<GROUPNAME>_average_task_results.csv
```
This will store the missing plots for Figure 10 B,C and D into the folder `experiments/data/figures/memory`.

