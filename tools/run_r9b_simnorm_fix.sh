#!/usr/bin/env bash
# ROUND-9b WM VALIDATION (2026-08-12): SimNorm anti-collapse fix attempt.
# ONLY changes vs round-9 (run_p09_simnorm, which regressed): the soft simplex
# now carries a UNIMIX floor (anti-dead-class) and simnorm_temp 1.0->0.5 (sharper
# = higher decoder contrast).  nl ALONE (WM validation) — judged on the P1/P2
# recon_loss trajectory (GATE: recon falls to ~0.08-0.10 like categorical, NOT
# the round-9 ~0.33 plateau) + the end-of-run WM transfer matrix / gains.
# If it clears, run test_sim next; else FALLBACK to categorical (pre-approved).
set -u
cd /home/koitkam/neural-APC-mbrl2
source /home/koitkam/neural-APC-mbrl2-env/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export DREAMER_ACT_HIST_REQUIRED=1
export DREAMER_DOB_ENABLED=1
export DREAMER_COMPILE=0

# ---- SimNorm anti-collapse fix ----
export DREAMER_RSSM_LATENT_TYPE=simnorm
export DREAMER_RSSM_SIMNORM_TEMP=0.5
export DREAMER_RSSM_JOINT_EMBED_COEF=0.5

STATUS=output/orchestrator_status.log
log(){ echo "[orchestrator] $(date '+%F %T') $*" | tee -a "$STATUS"; }

log "R9b START nonlinear WM-validation — SimNorm +unimix +τ0.5 +joint-embed 0.5; gain_match OFF"
DREAMER_GAIN_MATCH_COEF=0 python -m workflow.single_run \
    --simulation-dir simulation/nonlinear_sim \
    --out-dir output/nonlinear_sim/run_p09b_simnormfix
nl_rc=$?
log "R9b nonlinear WM-validation EXITED rc=${nl_rc}"
