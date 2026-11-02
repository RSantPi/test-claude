"""Run tuned GPR and KRR on several kinds of signal, and print the results.

Usage: python examples/compare_learners.py
Needs: pip install -e ".[eval]"
"""

import numpy as np

from smdata import SpectralMixture, effective_rank, step
from smdata.evaluate import learning_curve, make_gpr, make_krr

NOISE = 0.1
SIZES = [50, 200, 800]

CASES = {
    "smooth 2D (RBF-like)": (SpectralMixture([1.0], [[0.0, 0.0]], [[0.5, 0.5]]), None),
    "smooth 2D + step mean": (SpectralMixture([1.0], [[0.0, 0.0]], [[0.5, 0.5]]), step(1.0, [1, 0])),
    "one wave 2D": (SpectralMixture([1.0], [[1.5, 0.5]], [[0.25, 0.25]]), None),
    "two waves 2D": (
        SpectralMixture([0.5, 0.5], [[1.5, 0.5], [0.0, 2.5]], [[0.2, 0.2], [0.2, 0.2]]),
        None,
    ),
    "wide 3D": (SpectralMixture([1.0], [[0.0, 0.0, 0.0]], [[0.5, 0.5, 0.5]]), None),
}


def main():
    learners = {"GPR": make_gpr, "KRR": make_krr}
    header = f"{'n':>5} {'C':>7} {'ideal':>7} {'GPR':>7} {'KRR':>7} {'GPR cover':>10}"
    for name, (kernel, mean) in CASES.items():
        rank = effective_rank(kernel)
        print(f"\n{name}: r = {rank:.1f}, noise = {NOISE}")
        print(header)
        rows = learning_curve(kernel, SIZES, NOISE, learners, mean=mean, n_repeats=4, seed=1)
        for row in rows:
            print(
                f"{row['n']:>5} {row['complexity']:>7.4f} "
                f"{row['ideal_error']:>7.4f} {row['GPR_error']:>7.4f} "
                f"{row['KRR_error']:>7.4f} {row['GPR_coverage']:>10.3f}"
            )


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
