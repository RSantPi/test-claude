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
