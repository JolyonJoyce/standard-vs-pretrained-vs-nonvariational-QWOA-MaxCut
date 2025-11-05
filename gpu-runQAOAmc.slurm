#!/bin/sh
#SBATCH --job-name=nvQWOAmc-seed_$1
#SBATCH --account=pawsey0309-gpu
#SBATCH --export=NONE
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=1:00:00

module load python/3.11.6
module load gcc-native/14.2
module load py-numpy/2.1.2
# module load py-scikit-learn/
module load py-scipy/1.14.1

source $MYSCRATCH/qiskit-aer-venv-2.0.2/bin/activate

# Run and time the Python script

## Standard:
### 3-regular
# time srun -N 1 -n 1 -c 4 python -u run_QAOAmc.py -a std --nqubits 16 --PQCdepth 256 -q 500 -s $1 -G 3-reg -w True > $MYSOFTWARE/results/16_3R_STANDARD/seed_$1.txt
### Erdos-Renyi
# time srun -N 1 -n 1 -c 4 python -u run_QAOAmc.py -a std --nqubits 16 --PQCdepth 256 -q 500 -s $1 -G ER -e 0.3 -w True > $MYSOFTWARE/results/16_ER30_STANDARD/seed_$1.txt

## Pretrained:
### 3-regular
# time srun -N 1 -n 1 -c 4 python -u run_QAOAmc.py -a pt --nqubits 16 --PQCdepth 256 -q 500 -s $1 -G 3-reg -w True > $MYSOFTWARE/results/16_3R_PRETRAINED/seed_$1.txt
### Erdos-Renyi
# time srun -N 1 -n 1 -c 4 python -u run_QAOAmc.py -a pt --nqubits 16 --PQCdepth 256 -q 500 -s $1 -G ER -e 0.3 -w True > $MYSOFTWARE/results/16_ER30_PRETRAINED/seed_$1.txt

## Nonvariational:
### 3-regular
time srun -N 1 -n 1 --gpus=1 -p gpu python -u run_QAOAmc.py -a nv --nqubits 16 --PQCdepth 256 -q 500 -s $1 -G 3-reg -w True > $MYSOFTWARE/results/16_3R_NONVARIATIONAL/seed_$1.txt
### Erdos-Renyi
time srun -N 1 -n 1 --gpus=1 -p gpu python -u run_QAOAmc.py -a nv --nqubits 16 --PQCdepth 256 -q 500 -s $1 -G ER -e 0.3 -w True > $MYSOFTWARE/results/16_ER30_NONVARIATIONAL/seed_$1.txt
