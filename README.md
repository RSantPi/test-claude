# test-claude

A small repo for trying out working with Claude on GitHub.

## Dice roller

`roll.py` rolls dice using notation like `2d6+1`. It only needs Python 3.

```bash
python3 roll.py          # rolls 1d6
python3 roll.py 2d6+1    # rolls two six-sided dice and adds 1
python3 roll.py d20-2    # rolls one twenty-sided die and subtracts 2
```

Run the tests with:

```bash
python3 -m unittest
```

## Dataset generator

`smdata` generates regression datasets for testing Gaussian process regression
(GPR) and kernel ridge regression (KRR). The data comes from a Gaussian process
with a spectral mixture kernel, so the true model is known and every result has
a reference point.

```bash
pip install -e ".[dev]"
python -m pytest
```

```python
from smdata import SpectralMixture, make_dataset, ramp
from smdata.evaluate import make_gpr, make_krr, learning_curve

kernel = SpectralMixture.random(dim=2, n_components=3, seed=1)
data = make_dataset(kernel, n_train=500, noise=0.1, mean=ramp([1, 0]), seed=0)

data.X_train, data.y_train   # inputs in [-1, 1]^D, and noisy outputs
data.X_test, data.f_test     # test inputs, and the true signal there
data.rank                    # how much the kernel can express
data.complexity              # how much of the signal is out of reach here

rows = learning_curve(kernel, [50, 200, 800], noise=0.1,
                      learners={"GPR": make_gpr, "KRR": make_krr})
```

- **Domain:** inputs always lie in `[-1, 1]^D`, and the signal has variance 1.
  That gives length scales and noise levels the same meaning everywhere.
- **Mean function:** pass `constant`, `ramp`, `bowl`, `bump`, `step` or your own.
- **Fixed points:** `observed=(X, y)` forces the signal through chosen points.
- **Large datasets:** `method="features"` samples with random waves, which is
  approximate but avoids the cubic cost. Measuring `complexity` is still cubic,
  so also pass `measure_complexity=False`.

See [docs/complexity.md](docs/complexity.md) for the definitions, and
`examples/compare_learners.py` for a full comparison.
