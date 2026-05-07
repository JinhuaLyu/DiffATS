#!/bin/bash
# Watcher for the 4 ablation pipelines. Each pipeline has up to 3 jobs:
#   smoke   (1h)        : sanity-check the training script
#   full    (48h)       : the real 1000-epoch training
#   sample  (24h)       : 10000-sample generation, depends on full via afterok
#
# Per cycle (POLL_S):
#   smoke job:
#     - emit state changes
#     - on terminal: pass/fail by stdout (no Traceback + loss decreasing)
#     - if FAIL: scancel matching full + sample
#   full job:
#     - emit state changes
#     - on terminal: if final.pt exists anywhere under results_dir -> DONE for this slot
#       else -> resubmit full AND resubmit sample with afterok=new_full
#   sample job:
#     - emit state changes
#     - on terminal with success: count PNGs; if <10000 resubmit sample (script auto-resumes)
#
# Exits cleanly after LIMIT_S (~55 min) so a 1h Monitor can re-arm.

set -uo pipefail

POLL_S=${POLL_S:-900}
LIMIT_S=${LIMIT_S:-3300}
USER_NAME=$(whoami)
TARGET_SAMPLES=10000

declare -A SMOKE_JOB FULL_JOB SAMPLE_JOB FULL_SCRIPT SAMPLE_SCRIPT RESULTS_DIR
SMOKE_JOB[our_method]=16679842;        FULL_JOB[our_method]=16679843;        SAMPLE_JOB[our_method]=16679844
SMOKE_JOB[shared_bases]=16679845;      FULL_JOB[shared_bases]=16781043;      SAMPLE_JOB[shared_bases]=16781044
SMOKE_JOB[no_alignment]=16679848;      FULL_JOB[no_alignment]=16679849;      SAMPLE_JOB[no_alignment]=16679850
SMOKE_JOB[data_augmentation]=16679851; FULL_JOB[data_augmentation]=16679852; SAMPLE_JOB[data_augmentation]=16679853

FULL_SCRIPT[our_method]=${REPO_ROOT}/exps/celeba_hq/methods/our_method/job_train_full.sh
FULL_SCRIPT[shared_bases]=${REPO_ROOT}/exps/celeba_hq/methods/shared_bases/job_train_full.sh
FULL_SCRIPT[no_alignment]=${REPO_ROOT}/exps/celeba_hq/methods/no_alignment/job_train_full.sh
FULL_SCRIPT[data_augmentation]=${REPO_ROOT}/exps/celeba_hq/methods/data_augmentation/job_train_full.sh

SAMPLE_SCRIPT[our_method]=${REPO_ROOT}/exps/celeba_hq/methods/our_method/job_sample.sh
SAMPLE_SCRIPT[shared_bases]=${REPO_ROOT}/exps/celeba_hq/methods/shared_bases/job_sample.sh
SAMPLE_SCRIPT[no_alignment]=${REPO_ROOT}/exps/celeba_hq/methods/no_alignment/job_sample.sh
SAMPLE_SCRIPT[data_augmentation]=${REPO_ROOT}/exps/celeba_hq/methods/data_augmentation/job_sample.sh

ROOT=${DATA_ROOT}/ablation_results
RESULTS_DIR[our_method]=$ROOT/our_method
RESULTS_DIR[shared_bases]=$ROOT/shared_bases
RESULTS_DIR[no_alignment]=$ROOT/no_alignment
RESULTS_DIR[data_augmentation]=$ROOT/data_augmentation

METHODS=(our_method shared_bases no_alignment data_augmentation)

declare -A PREV
declare -A SMOKE_VERDICT
declare -A FULL_DONE         # 0/1 - final.pt seen
declare -A SAMPLE_DONE       # 0/1 - 10000 PNGs seen

job_active() {
  local j="$1"
  squeue -j "$j" -h -o "%T" 2>/dev/null \
    | awk '$1 != "CD" && $1 != "F" && $1 != "TO" && $1 != "CA" {print "active"; exit}'
}

job_terminal_state() {
  sacct -j "$1" -X -n -o State,Elapsed,ExitCode -P 2>/dev/null | head -1
}

emit() {
  local key="$1" msg="$2"
  if [ "${PREV[$key]:-}" != "$msg" ]; then
    echo "[$key] $msg"
    PREV[$key]="$msg"
  fi
}

smoke_classify() {
  local m="$1" j="$2"
  local out="${RESULTS_DIR[$m]}/logs/smoke_${j}.out"
  local err="${RESULTS_DIR[$m]}/logs/smoke_${j}.err"
  local n_err=0 n_out=0
  if [ -f "$err" ]; then n_err=$(grep -c "^Traceback" "$err" 2>/dev/null) || n_err=0; fi
  if [ -f "$out" ]; then n_out=$(grep -c "^Traceback" "$out" 2>/dev/null) || n_out=0; fi
  if [ "${n_err:-0}" -gt 0 ] || [ "${n_out:-0}" -gt 0 ]; then
    echo "FAIL_TRACEBACK"; return
  fi
  local latest_log
  latest_log=$(ls -t "${RESULTS_DIR[$m]}"/[0-9][0-9][0-9]-*/log.txt 2>/dev/null | head -1) || latest_log=""
  if [ -z "$latest_log" ] || [ ! -f "$latest_log" ]; then
    echo "INCONCLUSIVE no_log"; return
  fi
  # Read all "Train Loss: X.XXXX" or "loss=X.XXXX" matches in one shot.
  # Using mapfile + a single pipe avoids head -1 SIGPIPE under pipefail.
  local losses=() L
  while IFS= read -r L; do
    L="${L##*[: =]}"   # strip everything up through the last ':' '=' or ' '
    losses+=("$L")
  done < <(grep -oE "(Train Loss: |loss=)[0-9]+\.[0-9]+" "$latest_log" 2>/dev/null || true)
  if [ "${#losses[@]}" -eq 0 ]; then
    echo "INCONCLUSIVE no_loss"; return
  fi
  local first="${losses[0]}"
  local last="${losses[-1]}"
  if python3 -c "exit(0 if float('$last') < float('$first') else 1)" 2>/dev/null; then
    echo "PASS first=$first last=$last (via $(basename "$(dirname "$latest_log")"))"
  else
    echo "FAIL_NOT_DECREASING first=$first last=$last"
  fi
}

method_final_pt() { find "${RESULTS_DIR[$1]}" -name "final.pt" -type f 2>/dev/null | head -1; }
method_n_pngs() { ls ${RESULTS_DIR[$1]}/samples/images/*.png 2>/dev/null | wc -l; }

T_START=$(date +%s)
for m in "${METHODS[@]}"; do
  # All 4 smoke jobs already PASSED earlier in this experiment.
  # Hard-code verdict so the watcher won't re-classify off the now-much-larger
  # full-training log (which has plateaued loss and would falsely trip
  # FAIL_NOT_DECREASING).
  SMOKE_VERDICT[$m]="PASS (pre-recorded)"
  FULL_DONE[$m]=0
  SAMPLE_DONE[$m]=0
done

ALL_OK=0
while [ $ALL_OK -lt 4 ]; do
  ALL_OK=0
  for m in "${METHODS[@]}"; do
    # ---- smoke ----
    if [ -z "${SMOKE_VERDICT[$m]}" ]; then
      sj=${SMOKE_JOB[$m]}
      if [ -n "$(job_active $sj)" ]; then
        emit "$m:smoke" "job=$sj $(squeue -j $sj -h -o "%T|%M|%R" 2>/dev/null)"
      else
        v=$(smoke_classify "$m" "$sj")
        emit "$m:smoke" "TERMINAL job=$sj verdict=$v ($(job_terminal_state $sj))"
        SMOKE_VERDICT[$m]="$v"
        if [[ "$v" == FAIL* ]]; then
          fj=${FULL_JOB[$m]}; spj=${SAMPLE_JOB[$m]}
          [ -n "$(job_active $fj)" ] && scancel "$fj" 2>/dev/null && emit "$m:full" "CANCELLED ($fj) due to smoke FAIL"
          [ -n "$(job_active $spj)" ] && scancel "$spj" 2>/dev/null && emit "$m:sample" "CANCELLED ($spj) due to smoke FAIL"
          FULL_DONE[$m]=1; SAMPLE_DONE[$m]=1
        fi
      fi
    fi

    # ---- full ----
    if [ "${FULL_DONE[$m]}" -eq 0 ]; then
      fj=${FULL_JOB[$m]}
      finalp=$(method_final_pt "$m")
      if [ -n "$finalp" ]; then
        emit "$m:full" "DONE final.pt=$finalp"
        FULL_DONE[$m]=1
      elif [ -n "$(job_active $fj)" ]; then
        emit "$m:full" "job=$fj $(squeue -j $fj -h -o "%T|%M|%R" 2>/dev/null)"
      else
        ts=$(job_terminal_state "$fj")
        # Cancel orphan sample (DependencyNeverSatisfied)
        spj=${SAMPLE_JOB[$m]}
        [ -n "$(job_active $spj)" ] && scancel "$spj" 2>/dev/null
        new_full=$(sbatch --parsable "${FULL_SCRIPT[$m]}" 2>/dev/null)
        new_sample=$(sbatch --parsable --dependency=afterok:"$new_full" "${SAMPLE_SCRIPT[$m]}" 2>/dev/null)
        emit "$m:full" "TERMINAL[$fj]=$ts NO final.pt -> RESUBMIT full=$new_full sample=$new_sample"
        FULL_JOB[$m]="$new_full"
        SAMPLE_JOB[$m]="$new_sample"
      fi
    fi

    # ---- sample ----
    if [ "${SAMPLE_DONE[$m]}" -eq 0 ]; then
      spj=${SAMPLE_JOB[$m]}
      n_png=$(method_n_pngs "$m")
      if [ "$n_png" -ge "$TARGET_SAMPLES" ]; then
        emit "$m:sample" "DONE pngs=$n_png/$TARGET_SAMPLES"
        SAMPLE_DONE[$m]=1
      elif [ -n "$(job_active $spj)" ]; then
        emit "$m:sample" "job=$spj $(squeue -j $spj -h -o "%T|%M|%R" 2>/dev/null) pngs=$n_png/$TARGET_SAMPLES"
      else
        # Sample not running; only resubmit if final.pt is present (otherwise wait for full)
        finalp=$(method_final_pt "$m")
        if [ -n "$finalp" ]; then
          new_sample=$(sbatch --parsable "${SAMPLE_SCRIPT[$m]}" 2>/dev/null)
          emit "$m:sample" "TERMINAL[$spj] pngs=$n_png/$TARGET_SAMPLES -> RESUBMIT sample=$new_sample"
          SAMPLE_JOB[$m]="$new_sample"
        else
          emit "$m:sample" "no active sample job and no final.pt yet; waiting on full"
        fi
      fi
    fi

    [ "${SMOKE_VERDICT[$m]}" != "" ] && [ "${FULL_DONE[$m]}" -eq 1 ] && [ "${SAMPLE_DONE[$m]}" -eq 1 ] \
      && ALL_OK=$((ALL_OK+1))
  done

  now=$(date +%s)
  elapsed=$((now - T_START))
  if [ $elapsed -ge $LIMIT_S ]; then
    echo "[watcher_train] cycle limit ${LIMIT_S}s reached; clean exit"
    exit 0
  fi
  sleep "$POLL_S"
done

echo "ALL FOUR PIPELINES DONE (smoke + final.pt + 10000 samples)"
