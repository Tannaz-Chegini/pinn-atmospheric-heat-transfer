#!/bin/bash
# Submit all tasks of one season as a SLURM job array (one GPU per job, CHUNK tasks per job).
#
#   ACCOUNT=my-account PARTITION=gpu bash slurm/submit.sh summer
#   ACCOUNT=my-account PARTITION=gpu bash slurm/submit.sh winter
#
# Optional variables: CHUNK (tasks per job, default 16), VENV (Python environment,
# default ./.venv), FIRST/LAST (task range), SAVE_FIELDS=1 (also write .npz fields for maps).
# Finished tasks are skipped, so the same command can be repeated after failures.
set -eu
SEASON="${1:?usage: bash slurm/submit.sh summer|winter}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$HERE/.venv}"
CHUNK="${CHUNK:-16}"
: "${ACCOUNT:?set ACCOUNT to your SLURM account}"
: "${PARTITION:?set PARTITION to a GPU partition}"
mkdir -p "$HERE/logs"
source "$VENV/bin/activate"
N=$(python "$HERE/run.py" count --season "$SEASON" | tail -1)
FIRST="${FIRST:-0}"; LAST="${LAST:-$((N - 1))}"
NJOBS=$(( (LAST - FIRST + CHUNK) / CHUNK ))
EXTRA=""; [ "${SAVE_FIELDS:-0}" = 1 ] && EXTRA="--save-fields"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=pinn_${SEASON}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=0-12:00:00
#SBATCH --requeue
#SBATCH --array=0-$((NJOBS - 1))
#SBATCH --output=${HERE}/logs/${SEASON}_%A_%a.out
source "${VENV}/bin/activate"
cd "${HERE}"
a=\$(( ${FIRST} + SLURM_ARRAY_TASK_ID * ${CHUNK} ))
b=\$(( a + ${CHUNK} - 1 )); [ \$b -gt ${LAST} ] && b=${LAST}
for i in \$(seq \$a \$b); do
  python run.py task --season ${SEASON} --task \$i ${EXTRA}
done
EOF
echo "Submitted tasks ${FIRST}-${LAST} of season ${SEASON} as ${NJOBS} array jobs."
echo "Progress: ls results/raw/${SEASON}/*.csv | wc -l   (${N} when complete)"
