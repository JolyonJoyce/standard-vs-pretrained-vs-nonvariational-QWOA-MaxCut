To setup up a python virtual environment with GPU-enabled qiskit follow the instructions at:

https://github.com/PawseySC/qiskit_aer_rocm

Then, with the environment active, do:

	python -m pip install networkx

	cd <REPO_ROOT>/maxcut
	python -m pip install .

Test the installation with `qaoa_weighted_maxcut_v_1.0.py` in the root folder of this repository:

    srun -N 1 -n 1 --gpus=1 python qaoa_weighted_maxcut_v_1.0.py -D GPU
	
In your Slurm batch scripts include the following before using srun to start the simulation:

	. <PATH_TO_ACTIVATION_SCRIPT_FOLDER>/use-qiskit-source-rocm-setonix.sh
	
Take a look at the following page for example Slurm scripts for the Setonix GPU parition:

https://pawsey.atlassian.net/wiki/spaces/US/pages/51928618/Setonix+GPU+Partition+Quick+Start
