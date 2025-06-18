#!/bin/bash

source "config.sh"

cat singularity.def | sed "s|@SCRIPT_DIR@|${SCRIPT_DIR}|g" > singularity.def.run

# if result of last command is zero, then build
if [ $? -ne 0 ]; then
  echo "Error: Failed to create singularity.def.run from template."
  exit 1
fi

mkdir -p "${SINGULARITY_DIR}"

singularity build "${SINGULARITY_DIR}/structureiser.sif" "singularity.def.run"