# Iter-based, model-size-aware phase budget derivation for DreamerV4
# Returns total_steps, phase1_frac, phase2_frac, phase3_frac, and iter targets

def derive_phase_budgets(*, episode_length, complexity_score, model_size,
                        eps_per_iter_estimate=6, complexity_factor_cap=2.0):
    """
    Derives total_steps and phase fractions based on model_size, plant complexity, and episode_length.
    Anchored to paper and P52 evidence for iter targets.
    """
    # mbrl2 real-sim (2026-07-08): P1 raised for OBSERVER convergence.  p01
    # (run_20260707_realsim1) hit the P1 extension CAP WITHOUT plateauing (WM
    # next-state r 0.48, just under the 0.5 fidelity gate).  The WM(RSSM)+DOB
    # observer is the value-critical phase for the real-sim controller (LQG
    # separation — the actor acts on the observer's state estimate), so give it
    # room to cross the gate.
    P1_ITERS_BY_SIZE = {'S': 65, 'M': 82, 'L': 100}
    P2_ITERS_BY_SIZE = {'S': 25, 'M': 35, 'L': 45}
    # P3 RESTORED for the real-sim on-policy actor-critic (2026-07-08).  The
    # p121 reduction (S/M/L = 35/45/55) targeted the IMAGINATION actor, whose
    # slow late-P3 drift was an imagination + off-policy-REINFORCE artefact —
    # BOTH now removed (imagination deleted; the P3 actor trains on a dedicated
    # ON-POLICY buffer, not the seed replay buffer).  The real-sim actor learns
    # from REAL λ-returns and needs more P3 iters to converge after the critic
    # warmup.  (P3's env-step budget already maps to ~5× these nominal iters —
    # P3 collects ~1 ep/iter vs P1/P2's ~6 — so this is a comfortable ceiling;
    # the performance-aware P3 early-stop ends the phase once it has converged.)
    P3_ITERS_BY_SIZE = {'S': 45, 'M': 58, 'L': 72}
    factor = min(max(complexity_score / 4, 1.0), complexity_factor_cap)
    p1_iters = int(P1_ITERS_BY_SIZE[model_size] * factor)
    p2_iters = int(P2_ITERS_BY_SIZE[model_size] * factor)
    p3_iters = int(P3_ITERS_BY_SIZE[model_size] * factor)
    spi = eps_per_iter_estimate * episode_length
    p1 = p1_iters * spi
    p2 = p2_iters * spi
    p3 = p3_iters * spi
    total = p1 + p2 + p3
    return {
        'total_steps': int(total),
        'phase1_frac': p1 / total,
        'phase2_frac': p2 / total,
        'phase3_frac': p3 / total,
        'iter_targets': (p1_iters, p2_iters, p3_iters),
        'steps_per_iter': spi,
        'model_size': model_size,
        'complexity_score': complexity_score,
        'episode_length': episode_length,
        'source': 'auto:iter_based',
    }
