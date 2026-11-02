# Measuring complexity

Complexity here is measured, never set. It has two levels: what a kernel can
express, and how hard a particular dataset is. A third number, mismatch,
describes a learner on that dataset.

## Setup

Inputs lie in `[-1, 1]^D`, spread uniformly, and the signal has unit variance,
so `k(0) = 1`. The spectral mixture kernel, with weights summing to 1, is:

```
k(t) = sum_q w_q cos(2 pi mu_q . t) prod_p exp(-2 pi^2 t_p^2 v_qp)
```

The observed covariance is `K + noise^2 I`.

## Kernel: effective rank `r`

`r` counts the independent modes in functions drawn from the kernel. It depends
only on the kernel and the domain, not on sample size or noise.

```
r = tr(T)^2 / tr(T^2)
```

`T` is the kernel's integral operator on the domain. For this kernel `r` has a
closed form. Each axis contributes the integral

```
J(d, u) = integral from -2 to 2 of
          (2 - |t|) cos(2 pi d t) exp(-2 pi^2 u t^2) dt
```

which reduces to `erf` with a complex argument. Then:

```
r = 1 / sum_{q,q'} (w_q w_q' / 2) [ prod_p J(mu_qp - mu_q'p, u_p) / 4
                                  + prod_p J(mu_qp + mu_q'p, u_p) / 4 ]
u_p = v_qp + v_q'p
```

A constant function gives `r = 1`, and a pure wave gives `r = 2`. The code uses
the Faddeeva function, because a raw complex `erf` overflows for narrow peaks far
from zero. This form needs diagonal peak covariances and uniform inputs. For
other inputs, compute `matrix_rank(K)` instead.

## Dataset: complexity `C`

`C` is the share of the signal that even the true model can't recover. It's the
true GP's posterior variance, averaged over new uniform inputs:

```
C = mean over x of [ k(x, x) - k*^T (K + noise^2 I)^-1 k* ]
```

It runs from 0, fully determined, to 1, nothing learned. It depends on the
kernel, the training inputs and the noise, but not on the outputs. Averaged over
draws, it equals the true model's squared error on the signal, and a test checks
this.

Observed points act like extra training inputs with their own noise, so
conditioning lowers `C`. A mean function isn't part of the covariance, so it
doesn't change `C` or `r`.

`r` alone doesn't give `C`. A flat approximation,
`C = 1 / (1 + n / (r noise^2))`, was tested and removed because it was too low by
a factor of 7 to 40. The shape of the eigenvalue decay matters too.

## Learner: excess error

A learner's excess error is its squared error on the signal minus the true
model's, on the same data. This captures whatever `C` can't: a wrong kernel,
poor tuning, or a mean function the learner struggles with.

## Example results

From `examples/compare_learners.py`, with noise 0.1 and 4 repeats per row. Errors
are squared error on the signal, whose variance is 1. GPR and KRR use an RBF
kernel. Coverage is for GPR's 95% interval on noisy test outputs.

| Signal | `r` | `n` | `C` | Ideal | GPR | KRR | Coverage |
|---|---|---|---|---|---|---|---|
| Smooth 2D | 15.2 | 50 | 0.0621 | 0.0709 | 0.0771 | 0.1840 | 0.919 |
| | | 200 | 0.0063 | 0.0066 | 0.0066 | 0.0064 | 0.946 |
| | | 800 | 0.0012 | 0.0012 | 0.0012 | 0.0012 | 0.939 |
| Smooth 2D, step mean | 15.2 | 50 | 0.0621 | 0.0709 | 0.2316 | 0.2389 | 0.866 |
| | | 200 | 0.0063 | 0.0066 | 0.0612 | 0.0774 | 0.916 |
| | | 800 | 0.0012 | 0.0012 | 0.0206 | 0.0319 | 0.923 |
| One wave 2D | 9.3 | 50 | 0.0240 | 0.0228 | 0.4308 | 0.6222 | 0.921 |
| | | 200 | 0.0037 | 0.0035 | 0.0200 | 0.0522 | 0.954 |
| | | 800 | 0.0008 | 0.0009 | 0.0019 | 0.0022 | 0.940 |
| Two waves 2D | 13.4 | 50 | 0.0436 | 0.0383 | 0.6661 | 0.7338 | 0.928 |
| | | 200 | 0.0057 | 0.0059 | 0.0561 | 0.1139 | 0.946 |
| | | 800 | 0.0012 | 0.0012 | 0.0031 | 0.0055 | 0.942 |
| Wide 3D | 59.1 | 50 | 0.4836 | 0.4154 | 0.5144 | 0.4170 | 0.889 |
| | | 200 | 0.1148 | 0.1116 | 0.1126 | 0.1143 | 0.946 |
| | | 800 | 0.0130 | 0.0118 | 0.0120 | 0.0118 | 0.942 |

The ideal model knows the mean function, so the step changes only GPR and KRR.
