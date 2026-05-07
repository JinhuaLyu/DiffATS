#!/bin/bash
# Auto-resubmit watcher for the 3 preprocessing pipelines.
#
# For each slot, every POLL_S seconds:
#   1. If a job with the slot's preprocess name is currently active (PD/R/...)
#      in squeue -> continue, nothing to do.
#   2. Otherwise count finished shards in OUT_DIR.
#      If shard_count == EXPECTED:
#         - mark preprocess slot done; if stats output file(s) missing, resubmit stats.
#      Else:
#         - resubmit preprocess (script auto-resumes from existing shards).
#         - cancel any orphan stats job for this slot, sbatch new stats with
#           --dependency=afterok:<new_preprocess_id>.
#
# Emits one line per state change (Monitor turns each line into a notification).

set -uo pipefail

EXPECTED=60
POLL_S=${POLL_S:-900}   # 15 min
LIMIT_S=${LIMIT_S:-3300}  # ~55 min so Monitor 1h timeout doesn't kill mid-step

USER_NAME=$(whoami)
ROOT_OUT=/anvil/projects/x-eng260004/factor_diffusion/tucker_factors/celeba

declare -A PRE_NAME STATS_NAME PRE_SCRIPT STATS_SCRIPT OUT_DIR SHARD_GLOB STATS_FILES

# our_method
PRE_NAME[our_method]=celeba_p32r32_preprocess
STATS_NAME[our_method]=celeba_p32r32_stats
PRE_SCRIPT[our_method]=/home/x-jlyu5/jinhua/DiffATS/exps/celeba_hq/methods/our_method/job_preprocess.sh
STATS_SCRIPT[our_method]=/home/x-jlyu5/jinhua/DiffATS/exps/celeba_hq/methods/our_method/job_stats.sh
OUT_DIR[our_method]=$ROOT_OUT/our_method
SHARD_GLOB[our_method]="celebahq*procrustes_refimg*shard_*.pt"
STATS_FILES[our_method]="alpha_stats_procrustes_refimg_p32_r32.pt vhat_stats_procrustes_refimg_p32_r32.pt"

# shared_bases
PRE_NAME[shared_bases]=celeba_global_pca_preprocess
STATS_NAME[shared_bases]=celeba_global_pca_stats
PRE_SCRIPT[shared_bases]=/home/x-jlyu5/jinhua/DiffATS/exps/celeba_hq/methods/shared_bases/job_preprocess.sh
STATS_SCRIPT[shared_bases]=/home/x-jlyu5/jinhua/DiffATS/exps/celeba_hq/methods/shared_bases/job_stats.sh
OUT_DIR[shared_bases]=$ROOT_OUT/shared_bases
SHARD_GLOB[shared_bases]="celebahq*global_pca*shard_*.pt"
STATS_FILES[shared_bases]="alpha_stats_global_pca_p32_r32.pt"

# no_alignment
PRE_NAME[no_alignment]=celeba_no_align_preprocess
STATS_NAME[no_alignment]=celeba_no_align_stats
PRE_SCRIPT[no_alignment]=/home/x-jlyu5/jinhua/DiffATS/exps/celeba_hq/methods/no_alignment/job_preprocess.sh
STATS_SCRIPT[no_alignment]=/home/x-jlyu5/jinhua/DiffATS/exps/celeba_hq/methods/no_alignment/job_stats.sh
OUT_DIR[no_alignment]=$ROOT_OUT/no_alignment
SHARD_GLOB[no_alignment]="celebahq*no_alignment*shard_*.pt"
STATS_FILES[no_alignment]="alpha_stats_no_alignment_p32_r32.pt vhat_stats_no_alignment_p32_r32.pt"

SLOTS=(our_method shared_bases no_alignment)

declare -A PREV_LINE
T_START=$(date +%s)

active_job_id() {  # echo first active job id matching name, or empty
  local name="$1"
  squeue -u "$USER_NAME" -h -n "$name" -o "%i %T" 2>/dev/null \
    | awk '$2 != "CD" && $2 != "F" && $2 != "TO" && $2 != "CA" {print $1; exit}'
}

shard_count() {
  local slot="$1"
  ls ${OUT_DIR[$slot]}/${SHARD_GLOB[$slot]} 2>/dev/null | wc -l
}

stats_files_missing() {
  local slot="$1"
  for f in ${STATS_FILES[$slot]}; do
    [ -f "${OUT_DIR[$slot]}/$f" ] || { echo 1; return; }
  done
  echo 0
}

emit() {
  local slot="$1" msg="$2"
  if [ "${PREV_LINE[$slot]:-}" != "$msg" ]; then
    echo "[$slot] $msg"
    PREV_LINE[$slot]="$msg"
  fi
}

ALL_DONE=0
while [ $ALL_DONE -lt 3 ]; do
  ALL_DONE=0
  for slot in "${SLOTS[@]}"; do
    pre_name=${PRE_NAME[$slot]}
    stats_name=${STATS_NAME[$slot]}
    n=$(shard_count "$slot")
    pre_active=$(active_job_id "$pre_name")
    stats_active=$(active_job_id "$stats_name")

    if [ -n "$pre_active" ]; then
      pre_state=$(squeue -j "$pre_active" -h -o "%T|%M" 2>/dev/null)
      emit "$slot" "preprocess job=$pre_active state=$pre_state shards=$n/$EXPECTED"
      continue
    fi

    if [ "$n" -eq "$EXPECTED" ]; then
      missing=$(stats_files_missing "$slot")
      if [ "$missing" = "0" ]; then
        emit "$slot" "DONE shards=$n/$EXPECTED stats=present"
        ALL_DONE=$((ALL_DONE+1))
      else
        if [ -n "$stats_active" ]; then
          stats_state=$(squeue -j "$stats_active" -h -o "%T|%M|%R" 2>/dev/null)
          emit "$slot" "shards=$n/$EXPECTED stats job=$stats_active state=$stats_state"
        else
          new_stats=$(sbatch --parsable "${STATS_SCRIPT[$slot]}" 2>/dev/null)
          emit "$slot" "shards=$n/$EXPECTED stats missing -> RESUBMIT stats job=$new_stats"
        fi
      fi
    else
      [ -n "$stats_active" ] && scancel "$stats_active" 2>/dev/null
      new_pre=$(sbatch --parsable "${PRE_SCRIPT[$slot]}" 2>/dev/null)
      new_stats=$(sbatch --parsable --dependency=afterok:"$new_pre" "${STATS_SCRIPT[$slot]}" 2>/dev/null)
      emit "$slot" "INCOMPLETE shards=$n/$EXPECTED -> RESUBMIT preprocess=$new_pre stats=$new_stats (cancelled $stats_active)"
    fi
  done

  now=$(date +%s)
  elapsed=$((now - T_START))
  if [ $elapsed -ge $LIMIT_S ]; then
    echo "[watcher] cycle limit ${LIMIT_S}s reached; exit cleanly so Monitor can re-arm"
    exit 0
  fi
  sleep "$POLL_S"
done

echo "ALL THREE PIPELINES COMPLETE (preprocess + stats)"
