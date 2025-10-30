#!/bin/bash -l
#SBATCH --job-name=install-qiskit-source-rocm-setonix
#SBATCH --account=
#SBATCH --partition=gpu-dev
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --output=out-%x

# Source the environment script to ensure we have the right modules & venv
script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Remove any existing Aer source folder and Python virtual environment
rm -rf "$script_dir/qiskit-aer"
rm -rf "$script_dir/qiskit-aer-venv"

. "$script_dir/use-qiskit-source-rocm-setonix.sh"

if [ -z "$aer_ver" ]; then
    echo "Error: aer_ver is not set. Exiting."
    exit 1
fi

git clone https://github.com/Qiskit/qiskit-aer "$source_dir/qiskit-aer"
cd "$source_dir/qiskit-aer"

git checkout "$aer_ver"

# Install Aer's development requirements in the venv
python -m pip install -r requirements-dev.txt
python -m pip install pybind11

gcc_path="$(realpath "$(which gcc)")"
toolchain_path="$(dirname "$(dirname "$gcc_path")")"

python setup.py bdist_wheel -- \
  -DCMAKE_CXX_COMPILER=hipcc \
  -DCMAKE_CXX_FLAGS="--gcc-toolchain=$toolchain_path" \
  -DCMAKE_BUILD_TYPE=Release \
  -DAER_MPI=False \
  -DAER_THRUST_BACKEND=ROCM \
  -DAER_ROCM_ARCH=gfx90a \
  -DAER_DISABLE_GDR=True \
  --

python -m pip install dist/qiskit_aer*.whl

python -m pip install networkx

cd $script_dir/../maxcut
python -m pip install .
cd $script_dir
