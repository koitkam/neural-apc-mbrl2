#!/usr/bin/env bash
# ROUND-9 sequential GPU orchestrator (2026-08-11): SimNorm (TD-MPC2 soft
# simplicial latent) WM structural fix for the DV/CV gain-death, + joint-
# embedding (predict-next-latent) consistency (reconciliation B).  Runs the
# NONLINEAR WM-identification test FIRST (alone, full A10), then auto-restarts
# the test_sim regression the moment it finishes.
#
# ONLY change vs round-8 (run_nl_then_p09.sh): the categorical latent -> SimNorm
# (DREAMER_RSSM_LATENT_TYPE=simnorm) + joint-embedding (DREAMER_RSSM_JOINT_EMBED_
# COEF=0.5).  Everything else (BC-floor 0.1, reverted cap, gain_match off for nl)
# is held so the WM gain-death fix is attributable.
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
# Eager mode: a fresh torch.compile of the unrolled img_rollout stalls ~20 min
# single-threaded post-reboot.  Eager gives IDENTICAL results, just slower.
export DREAMER_COMPILE=0

# ---- ROUND-9 SimNorm WM structural fix ----
export DREAMER_RSSM_LATENT_TYPE=simnorm
export DREAMER_RSSM_SIMNORM_TEMP=1.0
export DREAMER_RSSM_JOINT_EMBED_COEF=0.5

STATUS=output/orchestrator_status.log
log(){ echo "[orchestrator] $(date '+%F %T') $*" | tee -a "$STATUS"; }

log "R9 START nonlinear run — SimNorm latent + joint-embed 0.5; gain_match OFF; BC-floor 0.1 + reverted cap"
DREAMER_GAIN_MATCH_COEF=0 python -m workflow.single_run \
    --simulation-dir simulation/nonlinear_sim \
    --out-dir output/nonlinear_sim/run_p09_simnorm
nl_rc=$?
log "R9 nonlinear run EXITED rc=${nl_rc}"

log "R9 START test_sim restart — SimNorm latent + joint-embed 0.5; BC-floor 0.1 + reverted cap"
python -m workflow.single_run \
    --simulation-dir simulation/test_sim \
    --out-dir output/test_sim/run_p17_simnorm
p9_rc=$?
log "R9 test_sim EXITED rc=${p9_rc} — orchestrator DONE (nl_rc=${nl_rc} p9_rc=${p9_rc})"
