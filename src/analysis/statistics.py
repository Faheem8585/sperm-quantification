"""Statistical analysis for comparing sperm populations."""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple


def compare_populations(
    population1: List[float],
    population2: List[float],
    test: str = 'mannwhitneyu'
) -> Dict[str, float]:
    """
    Compare two populations statistically.
    
    Args:
        population1: First population values.
        population2: Second population values.
        test: Statistical test ('mannwhitneyu', 'ttest', 'ks').
    
    Returns:
        Dictionary with test statistic and p-value.
    """
    if len(population1) < 2 or len(population2) < 2:
        return {'statistic': 0.0, 'p_value': 1.0}
    
    if test == 'mannwhitneyu':
        statistic, p_value = stats.mannwhitneyu(population1, population2)
    elif test == 'ttest':
        statistic, p_value = stats.ttest_ind(population1, population2)
    elif test == 'ks':
        statistic, p_value = stats.ks_2samp(population1, population2)
    else:
        raise ValueError(f"Unknown test: {test}")
    
    return {
        'statistic': float(statistic),
        'p_value': float(p_value)
    }


def compute_effect_size(
    population1: List[float],
    population2: List[float]
) -> float:
    """
    Compute Cohen's d effect size.
    
    Args:
        population1: First population.
        population2: Second population.
    
    Returns:
        Cohen's d value.
    """
    mean1 = np.mean(population1)
    mean2 = np.mean(population2)
    
    std1 = np.std(population1, ddof=1)
    std2 = np.std(population2, ddof=1)
    
    n1 = len(population1)
    n2 = len(population2)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    cohens_d = (mean1 - mean2) / pooled_std
    
    return float(cohens_d)


__all__ = ['compare_populations', 'compute_effect_size']
