#!/bin/bash -l

script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
. "$script_dir/use-qiskit-source-rocm-setonix.sh"

export MPICH_GPU_SUPPORT_ENABLED=1

wget https://github.com/mpi4py/mpi4py/releases/download/3.1.4/mpi4py-3.1.4.tar.gz
tar xfv mpi4py-3.1.4.tar.gz

cd mpi4py-3.1.4

cat << EOF > mpi.cfg
[mpi]
include_dirs		= /opt/cray/pe/mpich/8.1.27/ofi/gnu/9.1/include
library_dirs		= /opt/cray/pe/mpich/8.1.27/ofi/gnu/9.1/lib:/opt/cray/pe/mpich/8.1.27/gtl/lib
libraries		= mpi mpi_gtl_hsa
mpicc			= cc
mpicxx			= cc
EOF

python setup.py build --mpi=mpi
python -m pip install .
