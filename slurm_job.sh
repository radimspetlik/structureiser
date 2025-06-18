#!/bin/bash

source "./config.sh"

run_torch() {
  local sif_path="${SINGULARITY_DIR}/${SINGULARITY_IMAGE_NAME}.sif"

  if [ -f "${sif_path}" ]; then
    singularity exec --nv --cleanenv \
      -B "${SINGULARITY_DIR}:${SINGULARITY_DIR}" \
      "${sif_path}" \
      torchrun \
        --nnodes="${NNODES}" \
        --nproc-per-node="${GPUS_PER_NODE}" \
        --rdzv-id="${SLURM_JOB_ID}" \
        --rdzv-backend=c10d \
        --rdzv-endpoint="${MASTER_NODE}:29529" \
        ${ARGS}
  else
    pipenv run torchrun \
      --nnodes="${NNODES}" \
      --nproc-per-node="${GPUS_PER_NODE}" \
      --rdzv-id="${SLURM_JOB_ID}" \
      --rdzv-backend=c10d \
      --rdzv-endpoint="${MASTER_NODE}:29529" \
      ${ARGS}
  fi
}

cd "${SCRIPT_DIR}" || exit

if [ $# -eq 0 ]; then
  SCRIPT_NAME="structureiser.py"
  CONFIG_FILE="confs/lili.yml"
  ARGS="${SCRIPT_NAME} ${CONFIG_FILE}"
else
  ARGS="${@}"
fi

if [ -z "${SLURM_JOB_ID}" ]; then
  echo "Running interactively without SLURM"
  MASTER_NODE="localhost"
  NNODES=1
  GPUS_PER_NODE=1
  SLURM_JOB_ID="000000"
  run_torch
  exit 0
else
  echo "Running in SLURM job ${SLURM_JOB_ID}"
  NNODES=${SLURM_NNODES}
  GPUS_PER_NODE=$(scontrol show job ${SLURM_JOB_ID} | grep -oP 'gres/gpu:\K[^ ]+')
fi

export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3

# get node list and set MASTER_NODE/master_address
declare -A node_global_rank
node_list=$(scontrol show hostnames "${SLURM_NODELIST}")
index=0
for node in ${node_list[@]}; do
    node_global_rank["${node}"]=${index}
    index=$((index+1))
done

echo "node_list: ${node_list[@]}"

MASTER_NODE="$(scontrol show hostnames "${SLURM_NODELIST}" | head -1)"

for node in ${node_list[@]}; do
    echo "==> Launching on node: ${node}"
    run_torch
done
