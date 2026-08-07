#!/usr/bin/env bash
# Sequential GPU orchestrator (2026-08-03): run the NONLINEAR WM-identification
# test FIRST (alone, full A10 — fastest + safest), then auto-restart P09
# (test_sim regression) the moment it finishes.  Chosen over concurrent because
# two runs on one A10 halve each other's throughput (they fit in memory, but the
# priority nonlinear result would come ~2x slower).
#
# Each run writes its own in-process workflow.log; this script only appends its
# own transition markers to orchestrator_status.log (no duplicate training log).
set -u
cd /home/koitkam/neural-APC-mbrl2
source /home/koitkam/neural-APC-mbrl2-env/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export DREAMER_ACT_HIST_REQUIRED=1
export DREAMER_DOB_ENABLED=1
# Eager mode: the reboot wiped the /tmp Inductor cache, and a fresh torch.compile
# of the unrolled img_rollout (horizon 55/56) stalls ~20 min single-threaded.
# Eager gives IDENTICAL results, just slower per-iter — reliable for these tests.
export DREAMER_COMPILE=0

STATUS=output/orchestrator_status.log
log(){ echo "[orchestrator] $(date '+%F %T') $*" | tee -a "$STATUS"; }

log "START nonlinear run — gain_match OFF; p147 return_scale cap 500→8 + DV gain (ss_match+recon_cv)"
DREAMER_GAIN_MATCH_COEF=0 python -m workflow.single_run \
    --simulation-dir simulation/nonlinear_sim \
    --out-dir output/nonlinear_sim/run_p07_actorwmfix
nl_rc=$?
log "nonlinear run EXITED rc=${nl_rc}"

log "START P09 restart — test_sim; p147 return_scale cap 500→8 + DV gain (ss_match+recon_cv)"
python -m workflow.single_run \
    --simulation-dir simulation/test_sim \
    --out-dir output/test_sim/run_p15_actorwmfix
p9_rc=$?
log "P09 EXITED rc=${p9_rc} — orchestrator DONE (nl_rc=${nl_rc} p9_rc=${p9_rc})"
