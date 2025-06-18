#!/usr/bin/env bash

source "./config.sh"

bash "${SCRIPT_DIR}/slurm_job.sh" "evaluate.py"  ${CHECKPOINT_DIR} ${CHECKPOINT_FILENAME} ${OUTPUT_DIR} ${INPUT_DIR}