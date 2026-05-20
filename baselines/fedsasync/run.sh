#!/bin/bash
#SBATCH --job-name=FedSaSync
#SBATCH --output=logs/%A/%a.out
#SBATCH --error=logs/%A/%a.err
#SBATCH --cpus-per-task=12
#SBATCH --mem=12G
#SBATCH --time=1:00:00
#SBATCH --partition=galgo2
#SBATCH --array=0-149

# Trap de limpieza automática
cleanup_on_exit() {
    if [ -n "${WORKDIR}" ] && [ -d "${WORKDIR}" ]; then
        
        cd /tmp 2>/dev/null || true
        rm -rf "${WORKDIR}"
    fi
}
trap cleanup_on_exit EXIT

FS_VALUES=(0 1 2)
M_VALUES=(6 7 8 9 10)
REPS=(1 2 3 4 5 6 7 8 9 10)

FS_LEN=${#FS_VALUES[@]}
M_LEN=${#M_VALUES[@]}
I_LEN=${#REPS[@]}

TASK_ID=$SLURM_ARRAY_TASK_ID

fs_index=$(( TASK_ID / (M_LEN * I_LEN) ))
rest=$(( TASK_ID % (M_LEN * I_LEN) ))

m_index=$(( rest / I_LEN ))
i_index=$(( rest % I_LEN ))

fs=${FS_VALUES[$fs_index]}
m=${M_VALUES[$m_index]}
i=${REPS[$i_index]}

echo "fs=$fs m=$m i=$i"

if [ "$m" -eq 0 ]; then
    EXP_NAME="FedAvg"
else
    EXP_NAME="FedSaSync"
fi

JOB_TAG="${SLURM_ARRAY_JOB_ID:-$$}-${SLURM_ARRAY_TASK_ID}-${SLURM_PROCID}"
BACKUP_DIR="$HOME/fedsasync"
WORKDIR="/tmp/$(echo "${USER}" | tr '.' '-')-${JOB_TAG}"

FLWR_HOME="$WORKDIR/.flwr"
export FLWR_HOME="$FLWR_HOME"

rm -rf "$WORKDIR" 2>/dev/null
mkdir -p "$WORKDIR" || exit 1
mkdir -p "$FLWR_HOME" || exit 1

cp -r "$BACKUP_DIR/fedsasync" "$WORKDIR"
cp "$BACKUP_DIR/pyproject.toml" "$WORKDIR"
cp "$BACKUP_DIR/config.toml" "$FLWR_HOME/config.toml"
cd "$WORKDIR"

flwr run . --run-config "name=\"${EXP_NAME}\" semiasync-deg=${m} number-slow=${fs} exec-number=${i}" --stream

cp -r "$WORKDIR/results" "$BACKUP_DIR/"