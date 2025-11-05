#!/bin/bash --login
#SBATCH --partition=work
#SBATCH --account=pawsey0309
#SBATCH --export=NONE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=32000MB

module load python/3.11.6
module load gcc-native/14.2
module load py-numpy/2.1.2
# module load py-scikit-learn/
module load py-scipy/1.14.1

source $MYSCRATCH/qiskit-aer-venv-2.0.2/bin/activate

# Run and time the Python script

## Standard:
### 3-regular
# time srun -N 1 -n 1 -c 4 python -u run_QAOAmc.py -a std --nqubits 6 --PQCdepth 50 -q 50 -s $1 -G 3-reg -w True > $MYSOFTWARE/results/STD_TEST_3R/seed_$1.txt
### Erdos-Renyi
# time srun -N 1 -n 1 -c 4 python -u run_QAOAmc.py -a std --nqubits 6 --PQCdepth 50 -q 50 -s $1 -G ER -e 0.3 -w True > $MYSOFTWARE/results/STD_TEST_ER/seed_$1.txt

## Pretrained:
### 3-regular
# time srun -N 1 -n 1 -c 4 python -u run_QAOAmc.py -a pt --nqubits 6 --PQCdepth 50 -q 50 -s $1 -G 3-reg -w True > $MYSOFTWARE/results/PT_TEST_3R/seed_$1.txt
### Erdos-Renyi
# time srun -N 1 -n 1 -c 4 python -u run_QAOAmc.py -a pt --nqubits 6 --PQCdepth 50 -q 50 -s $1 -G ER -e 0.3 -w True > $MYSOFTWARE/results/PT_TEST_ER/seed_$1.txt

## Nonvariational:
### 3-regular
time srun -N 1 -n 1 -c 4 python -u run_QAOAmc.py -a nv --nqubits 6 --PQCdepth 50 -q 50 -s $1 -G 3-reg -w True > $MYSOFTWARE/results/NV_TEST_3R/seed_$1.txt
### Erdos-Renyi
time srun -N 1 -n 1 -c 4 python -u run_QAOAmc.py -a nv --nqubits 6 --PQCdepth 50 -q 50 -s $1 -G ER -e 0.3 -w True > $MYSOFTWARE/results/NV_TEST_ER/seed_$1.txt
