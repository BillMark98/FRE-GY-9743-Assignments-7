# Code reuse record for Assignment 7

**Summary: nothing was copied into HW7.** The repository ships the instructor's
solutions to every earlier assignment (HW1 interpolator, HW2–HW4 products and
model, HW5 gradients and fixed-cashflow engine, HW6 swaps, factory and engines).
Each was checked against my own earlier work and kept as shipped. The only code
written for HW7 is the `Interpolator1DMonotoneConvex` class body in
`fixedincomelib/utilities/numerics.py`, plus the notebook cells, tests and docs.

## Audit method

Every `.py` file under `fixedincomelib/` was compared byte-for-byte with its
counterpart in my HW6 working tree (`diff -rq`, excluding `__pycache__` and the
`screenshot/` images):

| status vs my HW6 tree | count | files |
|---|---|---|
| identical | 44 | everything not listed below, including `date/`, `data/`, `market/`, `model/`, `valuation/`, `yield_curve/yield_curve_model.py`, `yield_curve/model_builder.py`, `analytics/bond_calculator.py`, `apis/*` |
| changed | 5 | `utilities/numerics.py` and `utilities/__init__.py` (the HW7 task: new enum value, class stub, factory branch, export); `product/linear_products.py`, `product/product_factory.py`, `yield_curve/valuation_engine.py` (the instructor's HW6 solutions in place of mine) |
| only in one tree | 0 | — |

`static_files/` is identical.

## The earlier work HW7 sits on

### 1. HW1 interpolator — `Interpolator1DPCP`

HW7's PCP class is the instructor's vectorised version, unchanged since HW5. In
HW5 I compared it with my own HW1 implementation on 22,130 cases and found a
worst difference of exactly 0 (record in HW5's `docs/code-reuse.md`). The HW7
notebook's three PCP cells reproduce their supplied outputs exactly, which
re-confirms it. **Kept as shipped.**

The new class reuses nothing from it except two conventions: the FLAT-only
extrapolation check in the constructor and `searchsorted` for bucket lookup.
What it does reuse is the *design idea* of my HW1 solution — derive value and
gradient from one computation so they cannot drift — generalised to "dual rows"
(see `hw7-implementation.md` §6.2).

### 2. HW6 products, factory and engines — the instructor's solutions versus mine

HW7 ships the instructor's HW6 solutions. They differ from mine in structure, so
I checked them numerically with the HW7 notebook's three products priced on the
PCP curve in both repositories:

| product | my HW6 library | instructor's (HW7) | difference |
|---|---|---|---|
| FRA | 2,908.2866392380 | 2,908.2866392380 | 0 |
| RFR swap | 6,222.5175130702 | 6,222.5175130702 | 0 |
| basis swap | 8,351.3489081820 | 8,351.3489081820 | 0 |

Structural differences worth knowing, from the diff:

- **`ProductRFRSwap`**: the instructor rolls the *fixed* leg's accrual schedule
  on the payment holiday calendar, where I used the index's fixing calendar for
  both legs. For the notebook's inputs (USGS payment calendar, SOFR fixing
  calendar) the schedules coincide, hence the identical PVs; they could differ
  for an index whose fixing calendar has holidays the payment calendar lacks.
- **`ProductOvernightIndexBasisSwap`**: the two legs are built inline rather
  than through my `_overnight_leg` helper; argument for argument the same.
- **`ProductFactory`**: the instructor's spot-starting effective date is
  `on_index.valueDate(value_date)` — the index's own settlement lag — and a
  forward start is measured from that date. My HW6 builder took the value date
  itself as spot, which I had flagged as an assumption. The HW7 notebook calls
  the product constructors directly, so this is not exercised here; the
  instructor's reading is the one to keep.
- **Engines**: the same signed sums; only comments differ.

**Kept as shipped.** Nothing from my HW6 tree was carried over.

### 3. HW5 gradients and engine

Unchanged since HW6, where they were checked against my HW5 implementation on 23
probes with a worst difference of 8.9e−16. HW7's bump-and-reval test exercises
them again, now with the monotone convex interpolator underneath: the model-level
discount-factor gradient through the SOFR → SOFR-FLAT reference chain matches
finite differences to 1e−7. **Kept as shipped.**

### 4. HW2–HW4 products and model

Unchanged and not touched by HW7. The HW4 model builder's time convention
(ACT/ACT ISDA year fractions from the value date) is what the new interpolator's
`axis1` carries.

## Where the HW7 implementation reuses *supplied* HW7 code

- the `MONOTONE_CONVEX` enum value, the factory branch and the class skeleton
  (constructor signature, the four abstract methods) were supplied and are kept;
- `YieldCurveModelComponent.perturb_model_parameter` already rebuilds the
  interpolator after a bump, which is what makes the gradient test honest;
- the notebook's plot cell uses only public model API
  (`retrieve_model_component(...).state_data_interpolator`) and the two
  diagnostic properties added to the class.

## If a copy is ever needed

The solved HW7 interpolator file is self-contained. To carry it into a later
assignment that ships the stub again:

```bash
cp ../FRE-GY-9743-Assignments-7/fixedincomelib/utilities/numerics.py fixedincomelib/utilities/numerics.py
python tests/test_monotone_convex_interpolator.py
```
