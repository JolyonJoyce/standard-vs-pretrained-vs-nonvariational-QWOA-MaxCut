#!/bin/bash
#
# Define/Export common environment variables
export script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export source_dir="$script_dir/qiskit-rocm-src"
export aer_ver="4a5e830d0153b535cae55b1b3444d170865e1fd3"

# Versions of modules
py_ver="3.11.6"
pip_ver="23.1.2-py3.11.6"
st_ver="68.0.0-py3.11.6"
blas_ver="0.3.24"
rocm_ver="5.7.1"

# Load modules
module load python/$py_ver
module load py-pip/$pip_ver
module load py-setuptools/$st_ver
module load openblas/$blas_ver
module load rocm/$rocm_ver
module load craype-accel-amd-gfx90a
module load gcc

# Create a venv if it doesn't exist
if [ ! -d "$script_dir/qiskit-aer-venv" ]; then
  echo "Creating a new Python virtual environment at $script_dir/qiskit-aer-venv"
  python -m venv "$script_dir/qiskit-aer-venv"
fi

# Activate the virtual environment
source "$script_dir/qiskit-aer-venv/bin/activate"
