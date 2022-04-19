

networks=( microcircuit amorphous degreecontrolled degreecontrolled_no_io_specificity smallworld microcircuit_static microcircuit_random_dynamics )


idstart=1
idend=10

template="jobscript_template.sh"


echo ""
echo ""
echo "============================================================"
echo "                    Normal Tasks"
echo "============================================================"
echo ""

neuronnames=( hhneuron nonoiseneuron iafneuron )

for neuron in "${neuronnames[@]}"
do
	echo ""
	echo ""
	echo "======================== ${neuron} ========================"

	groupname="${neuron}"
	groupparam="--group_name ${groupname}"

	if [[ $neuron -eq "hhneuron" ]]
	then
		neuronparam=""
	elif [[ $neuron -eq "nonoiseneuron" ]]
	then
		neuronparam="--disable_conductance_noise"
	else
		neuronparam="--neuron_model iaf_cond_exp"
	fi
	
	taskparameters=( "" "--rate_tasks" )
	for taskparam in "${taskparameters[@]}"
	do
	
		for net in "${networks[@]}"
		do
			mkdir -p "${groupname}/${net}"
			netparam="--network_name ${net}"
			
			for runnr in $(seq $idstart $idend)
			do
				if [[ $taskparam -eq "" ]]
				then
					runname="${groupname}_${net}_spikes_${runnr}"
				else
					runname="${groupname}_${net}_rates_${runnr}"
				fi
	
				params="${netparam} ${groupparam} --runtitle run${runnr} ${taskparam} ${neuronparam}"
	
				echo $runname
				runscript="${groupname}/${net}/jobscript_${runname}.sh"
				cp $template $runscript
				sed -i "s|%%%NAME%%%|${runname}|g" $runscript
				sed -i "s|%%%PARAMS%%%|${params}|g" $runscript
			done
		done
	
	done
done



echo ""
echo ""
echo "============================================================"
echo "                    Memory Tasks"
echo "============================================================"
echo ""

groupname="memorytasks"
groupparam="--group_name ${groupname}"
taskparam="--step_duration 5 --max_delay 14"

for net in "${networks[@]}"
do
	mkdir -p "${groupname}/${net}"
	netparam="--network_name ${net}"
	
	for runnr in $(seq $idstart $idend)
	do

		params="${netparam} ${groupparam} --runtitle run${runnr} ${taskparam}"

		runname="${groupname}_${net}_${runnr}"
		echo $runname
		runscript="${groupname}/${net}/jobscript_${runname}.sh"
		cp $template $runscript
		sed -i "s|%%%NAME%%%|${runname}|g" $runscript
		sed -i "s|%%%PARAMS%%%|${params}|g" $runscript
	done
done

echo ""
echo ""
echo "============================================================"
echo "                 Different Training Trials"
echo "============================================================"
echo ""

groupname="different_steps"
groupparam="--group_name ${groupname}"
taskparam="--steps_per_trial 2 --step_duration 100. --input_dimension 4 --freeze_last_input"
diffstepsnetworks=( microcircuit amorphous )

trialnumbers=( 40 80 120 160 200 240 280 320 360 400 440 480 )

for trials in "${trialnumbers[@]}"
do
	trialparam="--train_trials ${trials}"

	for net in "${diffstepsnetworks[@]}"
	do
		mkdir -p "${groupname}/${net}"
		netparam="--network_name ${net}"
	
		for runnr in $(seq $idstart $idend)
		do
	
			params="${netparam} ${groupparam} --runtitle t${trials}_${runnr} ${taskparam} ${trialparam}"
	
			runname="${groupname}_${net}_t${trials}_${runnr}"
			echo $runname
			runscript="${groupname}/${net}/jobscript_${runname}.sh"
			cp $template $runscript
			sed -i "s|%%%NAME%%%|${runname}|g" $runscript
			sed -i "s|%%%PARAMS%%%|${params}|g" $runscript
		done
	done
done

echo ""
echo ""
echo "============================================================"
echo "                 Different Network Sizes"
echo "============================================================"
echo ""

groupname="different_N"
groupparam="--group_name ${groupname}"
diffnnetworks=( microcircuit amorphous )

networksizes=( 160 360 560 810 1000 2000 5000 10000 )

for N in "${networksizes[@]}"
do
	sizeparam="--N ${N}"

	for net in "${diffnnetworks[@]}"
	do
		mkdir -p "${groupname}/${net}"
		netparam="--network_name ${net}"
	
		for runnr in $(seq $idstart $idend)
		do
	
			params="${netparam} ${groupparam} --runtitle N${N}_${runnr} ${sizeparam}"
	
			runname="${groupname}_${net}_N${N}_${runnr}"
			echo $runname
			runscript="${groupname}/${net}/jobscript_${runname}.sh"
			cp $template $runscript
			sed -i "s|%%%NAME%%%|${runname}|g" $runscript
			sed -i "s|%%%PARAMS%%%|${params}|g" $runscript
		done
	done
done
