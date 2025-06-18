#!/usr/bin/env bash

source "./config.sh"

bash "${SCRIPT_DIR}/slurm_job.sh" "${SCRIPT_DIR}/train.py" "${SCRIPT_DIR}/confs/lili.yml"