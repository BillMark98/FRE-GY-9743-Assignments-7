"""Regression tests for Assignment 7 (monotone convex IFR interpolator).

The class under test is ``Interpolator1DMonotoneConvex`` in
``fixedincomelib/utilities/numerics.py``.  Beyond "it runs", the tests check the
properties the assignment asks for, and a few the design promises on top:

  * interpolates the node values and is continuous;
  * FLAT extrapolation on both wings; a one-node curve is constant;
  * monotone and convex on every node interval for the notebook's (compatible)
    data; bounded by the adjacent node values (no overshoot, positivity);
  * exact for constant, linear and (per-interval monotone) quadratic data;
  * integral identities: additivity, antisymmetry, the interval integral equals
    h_k * m_k, and agreement with brute-force quadrature;
  * sensible behaviour on incompatible data (hump, plateau, inflection,
    decreasing data);
  * the optional gradients match central finite differences, both at the
    interpolator and through the model's funding-curve chain;
  * the notebook's PCP and MONOTONE_CONVEX PVs, and the sign of their
    difference, explained by the forward rate each method implies.

Run either way::

    python tests/test_monotone_convex_interpolator.py
    pytest tests/test_monotone_convex_interpolator.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = next(p for p in Path(__file__).resolve().parents
             if (p / "fixedincomelib" / "utilities" / "numerics.py").is_file())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fixedincomelib import (  # noqa: E402
    qfCreateData1D, qfCreateDataCollection, qfCreateBuildMethod,
    qfCreateModelBuildMethodCollection, qfCreateModel, qfCreateProductFRA,
    qfCreateProductRFRSwap, qfCreateProductOvernightIndexBasisSwap,
    qfCreateValuationParameters, qfCreateValuationParametersCollection,
    qfCreateValueReport,
)
from fixedincomelib.date import Date, accrued  # noqa: E402
from fixedincomelib.utilities.numerics import (  # noqa: E402
    ExtrapMethod, InterpMethod, InterpolatorFactory,
    Interpolator1DMonotoneConvex, Interpolator1DPCP,
)

PCP, MC = "PIECEWISE_CONSTANT_LEFT_CONTINUOUS", "MONOTONE_CONVEX"

# ----------------------------------------------------------------------------
# The notebook's inputs
# ----------------------------------------------------------------------------
VALUE_DATE = "2026-01-02"
NODE_DATES = ["2026-04-02", "2026-10-02", "2027-07-02", "2028-01-03", "2029-01-02", "2031-01-02"]
SOFR = [0.025, 0.026, 0.028, 0.030, 0.035, 0.047]
FF = [0.023, 0.0237, 0.0252, 0.0265, 0.0305, 0.0405]
NODE_TIMES = np.array([accrued(Date(VALUE_DATE), Date(d)) for d in NODE_DATES])

# Notebook outputs: PCP from the supplied notebook, MONOTONE_CONVEX from this implementation.
NOTEBOOK_PV = {
    "FRA": {PCP: 2908.2866392379638, MC: 3262.3770589046},
    "RFR swap": {PCP: 6222.517513070154, MC: 12860.3995181355},
    "Basis swap": {PCP: 8351.348908181993, MC: 7624.0991106551},
}


def mc(x, f):
    return InterpolatorFactory.create_1d_interpolator(
        np.asarray(x, float), np.asarray(f, float), InterpMethod.MONOTONE_CONVEX, ExtrapMethod.FLAT)


def build_model(method):
    dc = qfCreateDataCollection([
        qfCreateData1D("INSTANTANEOUS FORWARD RATE", "SOFR-1B-IFR", pd.DataFrame({"values": SOFR}, index=NODE_DATES)),
        qfCreateData1D("INSTANTANEOUS FORWARD RATE", "FF-1B-IFR", pd.DataFrame({"values": FF}, index=NODE_DATES)),
        qfCreateData1D("INSTANTANEOUS FORWARD RATE", "SOFR-1B-FUNDING-IFR",
                       pd.DataFrame({"values": [0.0005]}, index=["2027-01-04"])),
    ])
    inputs = [
        ("YIELD_CURVE_INDEX", {"TARGET": "SOFR-1B", "INSTANTANEOUS FORWARD RATE": "SOFR-1B-IFR"}),
        ("YIELD_CURVE_INDEX", {"TARGET": "FF-1B", "INSTANTANEOUS FORWARD RATE": "FF-1B-IFR"}),
        ("YIELD_CURVE_FUNDING", {"TARGET": "SOFR-1B-FLAT", "REFERENCE INDEX": "SOFR-1B",
                                 "INSTANTANEOUS FORWARD RATE": "SOFR-1B-FUNDING-IFR"}),
    ]
    bms = qfCreateModelBuildMethodCollection([
        qfCreateBuildMethod(k, {**i, "INTERPOLATION METHOD": method, "EXTRAPOLATION METHOD": "FLAT"})
        for k, i in inputs])
    return qfCreateModel(VALUE_DATE, "YIELD_CURVE", dc, bms)


def notebook_products():
    fra = qfCreateProductFRA(effective_date='2026-05-05', term_or_termination_date='2026-11-05',
                             projection_index='SOFR-1B', fixed_rate=0.032, notional=1000000,
                             pay_or_rec='receive', accrual_basis='ACT/360',
                             business_day_convention='F', holiday_convention='USGS')
    swap = qfCreateProductRFRSwap(effective_date='2026-04-02', term_or_termination_date='2029-04-02',
                                  on_index='SOFR-1B', fixed_rate=0.034, notional=1000000,
                                  pay_or_rec='receive', accrual_period='1Y',
                                  floating_leg_accrual_period='1Y', accrual_basis='ACT/360',
                                  compounding_method='compound', spread=0.0, payment_off_set='2D',
                                  pay_business_day_convention='F', pay_holiday_convention='USGS')
    basis = qfCreateProductOvernightIndexBasisSwap(
        effective_date='2026-07-02', term_or_termination_date='2028-07-03',
        on_index_leg_1='SOFR-1B', on_index_leg_2='FF-1B', spread_over_leg_1=0.001,
        notional=1000000, pay_or_rec_leg_1='receive', accrual_period_1='3M',
        accrual_period_2='3M', accrual_basis='ACT/360', compounding_method='compound',
        payment_off_set='2D', pay_business_day_convention='F', pay_holiday_convention='USGS')
    return {"FRA": fra, "RFR swap": swap, "Basis swap": basis}


def pv(model, product):
    params = qfCreateValuationParametersCollection([
        qfCreateValuationParameters("FUNDING INDEX PARAMETER", {"FUNDING INDEX": "SOFR-1B-FLAT"})])
    return qfCreateValueReport(model, product, params, "PV")[0][1]


def grid_values(it, lo, hi, n=20001):
    g = np.linspace(lo, hi, n)
    return g, np.array([it.interpolate(t) for t in g])


# ----------------------------------------------------------------------------
# Interpolation properties
# ----------------------------------------------------------------------------
def test_passes_through_nodes_and_is_continuous():
    it = mc(NODE_TIMES, SOFR)
    assert isinstance(it, Interpolator1DMonotoneConvex)
    for t, v in zip(NODE_TIMES, SOFR):
        assert abs(it.interpolate(t) - v) < 1e-15
        # left and right limits agree: the two adjacent pieces both pass through the node
        assert abs(it.interpolate(t - 1e-12) - it.interpolate(t + 1e-12)) < 1e-12


def test_flat_extrapolation_and_single_node_is_constant():
    it = mc(NODE_TIMES, SOFR)
    assert it.interpolate(0.0) == SOFR[0] and it.interpolate(NODE_TIMES[0] / 2) == SOFR[0]
    assert it.interpolate(NODE_TIMES[-1] + 10.0) == SOFR[-1]
    assert abs(it.integrate(0.0, NODE_TIMES[0]) - SOFR[0] * NODE_TIMES[0]) < 1e-15
    # the one-node funding curve must be constant
    one = mc([1.0], [0.0005])
    assert all(one.interpolate(t) == 0.0005 for t in (0.0, 0.5, 1.0, 7.0))
    assert abs(one.integrate(0.0, 3.0) - 0.0015) < 1e-15
    assert np.allclose(one.gradient_of_integrated_value_wrt_ordinate(0.0, 3.0), [3.0])


def test_two_nodes_is_the_straight_line():
    it = mc([1.0, 3.0], [0.02, 0.03])
    for t in np.linspace(1.0, 3.0, 11):
        assert abs(it.interpolate(t) - (0.02 + 0.005 * (t - 1.0))) < 1e-15
    assert abs(it.integrate(1.0, 3.0) - 0.05) < 1e-15


def test_reproduces_constant_linear_and_quadratic_data_exactly():
    x = np.array([0.5, 1.0, 1.75, 2.5, 4.0, 6.0])
    for a, b, c in ((0.03, 0.0, 0.0), (0.02, 0.004, 0.0), (0.02, 0.003, 0.0005)):
        poly = lambda t: a + b * t + c * t * t
        it = mc(x, poly(x))
        for t in np.linspace(x[0], x[-1], 97):
            assert abs(it.interpolate(t) - poly(t)) < 1e-14
        exact = lambda s, e: a * (e - s) + b * (e ** 2 - s ** 2) / 2 + c * (e ** 3 - s ** 3) / 3
        assert abs(it.integrate(x[0], x[-1]) - exact(x[0], x[-1])) < 1e-14
        assert abs(it.integrate(0.8, 5.3) - exact(0.8, 5.3)) < 1e-14
        assert np.allclose(it.interval_curvatures, 2 * c, atol=1e-13)


def test_monotone_and_convex_on_the_notebook_data():
    for data in (SOFR, FF):
        it = mc(NODE_TIMES, data)
        x = NODE_TIMES
        chord_mid = (np.array(data[:-1]) + np.array(data[1:])) / 2
        assert np.all(it.interval_curvatures > 0)                       # convex pieces
        assert np.all(it.interval_averages < chord_mid)                 # Hermite-Hadamard: below chord midpoints
        for k in range(len(x) - 1):
            g, v = grid_values(it, x[k], x[k + 1], 4001)
            assert np.all(np.diff(v) >= -1e-16)                           # monotone within the piece
            assert np.all(np.diff(v, 2) >= -1e-16)                        # convex within the piece
            assert v.min() >= data[k] - 1e-15 and v.max() <= data[k + 1] + 1e-15   # no overshoot
        g, v = grid_values(it, 0.0, x[-1] + 1.0)
        assert np.all(np.diff(v) >= -1e-16) and v.min() > 0.0            # monotone and positive overall


def test_integral_identities_and_quadrature():
    it = mc(NODE_TIMES, SOFR)
    x, h = NODE_TIMES, np.diff(NODE_TIMES)
    for k in range(len(x) - 1):                                           # piece integral = h_k * m_k
        assert abs(it.integrate(x[k], x[k + 1]) - h[k] * it.interval_averages[k]) < 1e-15
    a, b, c = 0.1, 1.9, 5.7
    assert abs(it.integrate(a, c) - it.integrate(a, b) - it.integrate(b, c)) < 1e-15   # additive
    assert abs(it.integrate(c, a) + it.integrate(a, c)) < 1e-15                        # antisymmetric
    assert it.integrate(b, b) == 0.0
    g, v = grid_values(it, a, c, 200001)
    assert abs(np.trapezoid(v, g) - it.integrate(a, c)) < 1e-9                         # brute force
    # windows that START and END strictly inside pieces exercise the antiderivative away from u = 0, 1
    for lo, hi in ((0.4, 0.6), (0.9, 2.3), (1.7, 4.2)):
        g, v = grid_values(it, lo, hi, 200001)
        assert abs(np.trapezoid(v, g) - it.integrate(lo, hi)) < 1e-9


def test_incompatible_data_is_handled_without_overshoot():
    x = np.arange(4.0)
    hump = mc(x, [0.02, 0.03, 0.02, 0.01])                 # maximum at a node
    assert np.all(hump.interval_curvatures <= 0)           # concave pieces, no invented extremum
    g, v = grid_values(hump, 0.0, 3.0)
    assert v.max() <= 0.03 + 1e-15 and v.min() >= 0.01 - 1e-15
    plateau = mc(x, [0.02, 0.03, 0.03, 0.04])              # equal neighbours -> flat piece
    assert plateau.interval_curvatures[1] == 0.0
    assert all(plateau.interpolate(t) == 0.03 for t in np.linspace(1.0, 2.0, 7))
    bend = mc(np.arange(5.0), [0.01, 0.02, 0.04, 0.05, 0.055])   # inflection -> straight piece
    assert bend.interval_curvatures[1] == 0.0
    assert abs(bend.interpolate(1.5) - 0.03) < 1e-15
    down = mc(NODE_TIMES, list(reversed(SOFR)))            # decreasing data -> decreasing curve
    g, v = grid_values(down, 0.0, 6.0)
    assert np.all(np.diff(v) <= 1e-16)
    # the cap binds exactly when the quadratic would stop being monotone: zero end slope, no overshoot
    steep = mc([0.0, 1.0, 2.0], [0.0, 0.01, 0.1])
    g, v = grid_values(steep, 0.0, 1.0)
    assert np.all(np.diff(v) >= -1e-16) and abs(v[0] - 0.0) < 1e-15


def test_gradients_match_central_finite_differences():
    f = np.array(SOFR)
    it = mc(NODE_TIMES, f)
    eps, worst = 1e-7, 0.0
    for t in (0.1, 0.5, 1.0, 1.7, 2.5, 4.0, 6.0):
        g_val = it.gradient_wrt_ordinate(t)
        g_int = it.gradient_of_integrated_value_wrt_ordinate(0.0, t)
        assert abs(g_int.sum() - t) < 1e-12                     # shifting all nodes by 1 adds t
        for j in range(len(f)):
            up, dn = f.copy(), f.copy()
            up[j] += eps
            dn[j] -= eps
            worst = max(worst,
                        abs((mc(NODE_TIMES, up).interpolate(t) - mc(NODE_TIMES, dn).interpolate(t)) / (2 * eps) - g_val[j]),
                        abs((mc(NODE_TIMES, up).integrate(0.0, t) - mc(NODE_TIMES, dn).integrate(0.0, t)) / (2 * eps) - g_int[j]))
    assert worst < 1e-8


# ----------------------------------------------------------------------------
# Through the model
# ----------------------------------------------------------------------------
def test_build_method_selects_the_class_and_funding_curve_is_constant():
    model = build_model(MC)
    for name in ("SOFR-1B", "FF-1B", "SOFR-1B-FLAT"):
        assert isinstance(model.retrieve_model_component(name).state_data_interpolator,
                          Interpolator1DMonotoneConvex)
    assert isinstance(build_model(PCP).retrieve_model_component("SOFR-1B").state_data_interpolator,
                      Interpolator1DPCP)
    fund = model.retrieve_model_component("SOFR-1B-FLAT")
    for d in ("2026-03-01", "2027-01-04", "2030-06-30"):
        t = accrued(Date(VALUE_DATE), Date(d))
        assert abs(fund.discount_factor(Date(d)) - np.exp(-0.0005 * t)) < 1e-15


def test_model_discount_factor_gradient_bump_and_reval():
    model = build_model(MC)
    for date in ("2026-07-01", "2027-11-15", "2032-01-02"):
        expiry = Date(date)
        grad = []
        model.discount_factor_gradient_wrt_state("SOFR-1B-FLAT", expiry, grad)
        base = model.discount_factor("SOFR-1B-FLAT", expiry)
        for name in ("SOFR-1B", "SOFR-1B-FLAT"):
            block = grad[model.retrieve_model_component_index(name)]
            for j in range(len(block)):
                model.perturb_model_parameter(name, j, 1e-6)
                up = model.discount_factor("SOFR-1B-FLAT", expiry)
                model.perturb_model_parameter(name, j, -2e-6)
                dn = model.discount_factor("SOFR-1B-FLAT", expiry)
                model.perturb_model_parameter(name, j, 1e-6)
                assert abs((up - dn) / 2e-6 - block[j]) < 1e-7
        assert abs(model.discount_factor("SOFR-1B-FLAT", expiry) - base) < 1e-15   # restored


def test_notebook_pvs_and_why_they_differ():
    products = notebook_products()
    models = {PCP: build_model(PCP), MC: build_model(MC)}
    for name, product in products.items():
        for method, model in models.items():
            assert abs(pv(model, product) - NOTEBOOK_PV[name][method]) < 1e-6, (name, method)

    # The FRA receives fixed 3.2% against the SOFR forward over [2026-05-05, 2026-11-05].
    # PCP puts 2.6% (the Oct-2026 node, left-continuous) on that whole window; the
    # monotone convex curve rises from 2.5% at Apr-2026 towards 2.6%, so its forward
    # is lower and the receiver's PV is higher.
    def forward(model):
        d0, d1 = model.discount_factor("SOFR-1B", Date("2026-05-05")), model.discount_factor("SOFR-1B", Date("2026-11-05"))
        alpha = (Date("2026-11-05").serialNumber() - Date("2026-05-05").serialNumber()) / 360.0
        return (d0 / d1 - 1.0) / alpha
    assert forward(models[MC]) < forward(models[PCP])
    assert NOTEBOOK_PV["FRA"][MC] > NOTEBOOK_PV["FRA"][PCP]
    # Same direction for the receive-fixed swap (lower projected floating leg);
    # the basis swap receives SOFR, so the lower SOFR forwards reduce its PV.
    assert NOTEBOOK_PV["RFR swap"][MC] > NOTEBOOK_PV["RFR swap"][PCP]
    assert NOTEBOOK_PV["Basis swap"][MC] < NOTEBOOK_PV["Basis swap"][PCP]


def test_mc_pcp_agree_when_both_are_constant():
    # Flat data: both methods give the same flat curve, so every PV coincides.
    flat_nodes = pd.DataFrame({"values": [0.03] * 6}, index=NODE_DATES)

    def flat_model(method):
        dc = qfCreateDataCollection([
            qfCreateData1D("INSTANTANEOUS FORWARD RATE", "SOFR-1B-IFR", flat_nodes),
            qfCreateData1D("INSTANTANEOUS FORWARD RATE", "FF-1B-IFR", flat_nodes),
            qfCreateData1D("INSTANTANEOUS FORWARD RATE", "SOFR-1B-FUNDING-IFR",
                           pd.DataFrame({"values": [0.0005]}, index=["2027-01-04"]))])
        inputs = [
            ("YIELD_CURVE_INDEX", {"TARGET": "SOFR-1B", "INSTANTANEOUS FORWARD RATE": "SOFR-1B-IFR"}),
            ("YIELD_CURVE_INDEX", {"TARGET": "FF-1B", "INSTANTANEOUS FORWARD RATE": "FF-1B-IFR"}),
            ("YIELD_CURVE_FUNDING", {"TARGET": "SOFR-1B-FLAT", "REFERENCE INDEX": "SOFR-1B",
                                     "INSTANTANEOUS FORWARD RATE": "SOFR-1B-FUNDING-IFR"})]
        bms = qfCreateModelBuildMethodCollection([
            qfCreateBuildMethod(k, {**i, "INTERPOLATION METHOD": method, "EXTRAPOLATION METHOD": "FLAT"})
            for k, i in inputs])
        return qfCreateModel(VALUE_DATE, "YIELD_CURVE", dc, bms)
    m_pcp, m_mc = flat_model(PCP), flat_model(MC)
    for product in notebook_products().values():
        assert abs(pv(m_pcp, product) - pv(m_mc, product)) < 1e-9


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception as exc:                     # assertion OR unexpected error
            failed += 1
            print(f"FAIL  {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
