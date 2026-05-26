# Iter-based, model-size-aware phase budget derivation for DreamerV4
# Returns total_steps, phase1_frac, phase2_frac, phase3_frac, and iter targets

def derive_phase_budgets(*, episode_length, complexity_score, model_size,
                        eps_per_iter_estimate=6, complexity_factor_cap=2.0):
    """
    Derives total_steps and phase fractions based on model_size, plant complexity, and episode_length.
    Anchored to paper and P52 evidence for iter targets.
    """
    P1_ITERS_BY_SIZE = {'S': 50, 'M': 65, 'L': 80}
    P2_ITERS_BY_SIZE = {'S': 25, 'M': 35, 'L': 45}
    P3_ITERS_BY_SIZE = {'S': 50, 'M': 70, 'L': 90}
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
