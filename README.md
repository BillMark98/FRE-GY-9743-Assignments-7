# FRE-GY-9743-Assignments-2026fall
Assignments for NYU, FRE-GY 9743, Fall 2026.

## HW7 — implementation notes

`Interpolator1DMonotoneConvex` (in `fixedincomelib/utilities/numerics.py`) is
implemented: a Hagan–West-style piecewise-quadratic IFR interpolator that passes
through the supplied points and is monotone and convex on every node interval for
compatible data, with FLAT extrapolation and the optional gradient interface.
The notebook's plot and comparison cells are filled in. Documentation lives in
[`docs/`](docs/):

- [docs/hw7-implementation.md](docs/hw7-implementation.md) — the reading of the
  task, the method and its proofs, how incompatible data is handled, the code
  design, the results on the notebook inputs with the PV differences explained,
  and the verification (tests, quadrature, bump-and-reval, mutation testing).
- [docs/hw7_ifr_curves.png](docs/hw7_ifr_curves.png) — the IFR curves under both
  methods.
- [docs/code-reuse.md](docs/code-reuse.md) — audit of earlier work: nothing
  needed copying; the shipped HW6 solutions were checked against mine.

Theoretical background: [`docs/stochastic_calculus_and_yield_curve_calibration.md`](../../docs/stochastic_calculus_and_yield_curve_calibration.md) (§7).

Run the regression tests with:

```bash
python tests/test_monotone_convex_interpolator.py
```
