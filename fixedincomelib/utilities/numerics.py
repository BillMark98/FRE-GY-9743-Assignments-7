import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'
    MONOTONE_CONVEX = 'MONOTONE_CONVEX'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        return float(self.values_[idx])

    def integrate(self, start_x: float, end_x: float) -> float:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        overlaps = np.maximum(hi - lo, 0.0)
        return float(np.dot(np.asarray(self.values_, dtype=float), overlaps))

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        grad = np.zeros(self.length_)
        grad[idx] = 1.0
        return grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        return np.maximum(hi - lo, 0.0)



class Interpolator1DMonotoneConvex(Interpolator1D):
    """HW7: a monotone, convex IFR interpolator through the supplied node values.

    Inputs
    ------
    ``axis1``  : node times t_0 < t_1 < ... < t_{n-1}      (years from value date)
    ``values`` : the IFR *at* those times, f_j := f(t_j).  The curve passes
                 through every (t_j, f_j).  (Contrast with ``Interpolator1DPCP``,
                 which treats values[j] as the constant rate ON (t_{j-1}, t_j].)

    Construction (Hagan & West, "Interpolation Methods for Curve Construction",
    AMF 2006, eqs. 34-48, adapted to point data)
    -------------------------------------------------------------------------
    On each node interval I_k = [t_k, t_{k+1}] of length h_k, with the local
    coordinate u = (t - t_k)/h_k, the curve is Hagan-West's quadratic

        f(t) = m_k + g0_k (1 - 4u + 3u^2) + g1_k (-2u + 3u^2),          (HW 47)
        g0_k = f_k - m_k,   g1_k = f_{k+1} - m_k,

    which is the UNIQUE quadratic that (a) equals f_k at u=0, (b) equals
    f_{k+1} at u=1 and (c) has interval average m_k, because the two basis
    functions integrate to zero over [0, 1].  Hagan-West obtain m_k from a
    market discrete forward; here the data are point values, so m_k is the one
    free parameter per interval and is chosen as follows:

        m_k = (f_k + f_{k+1}) / 2  -  h_k^2 c_k / 12,                    (*)

    i.e. the exact average of a quadratic with second derivative c_k through
    the two endpoints (Hermite-Hadamard: the average of a convex function lies
    below the chord midpoint).  Indeed f'' == c_k on I_k, so c_k IS the
    curvature of the piece.  c_k is a local, shape-limited estimate:

      1. c_L = second derivative of the parabola through nodes k-1, k, k+1
         (= 2 x the second divided difference), c_R likewise through k, k+1, k+2;
      2. combine: if c_L and c_R have opposite signs the data shows an
         inflection, so c_k = 0 (a straight piece); otherwise c_k is their
         mean (one-sided at the two end intervals; 0 with only two nodes);
      3. cap  |c_k| <= 2 |delta_k| / h_k,   delta_k = (f_{k+1} - f_k)/h_k.

    Why the cap: with e := h_k^2 c_k/12 the piece's end slopes are
    f'(t_k+) = (delta_k - 6e/h_k) and f'(t_{k+1}-) = (delta_k + 6e/h_k), so the
    quadratic is monotone on I_k iff |e| <= |delta_k|/6 iff |c_k| <= 2|delta_k|/h_k
    (Hagan-West's sector (i): g'(0) and g'(1) share a sign).  Capping, rather
    than inserting Hagan-West's flat segments (sectors ii-iv), keeps every
    piece a single quadratic; the price is a bound on the curvature a piece
    can carry, which no realistic IFR data approaches (see docs/hw7-*.md).

    Guarantees
    ----------
    * interpolating and continuous (C^0 at nodes, C^infinity inside pieces);
    * each piece is monotone (so bounded by its two node values: no overshoot,
      and positive data give a positive curve);
    * each piece has a single curvature sign (a quadratic cannot inflect);
      with convex data (non-decreasing secant slopes) every piece is convex,
      with concave data every piece is concave, at a data inflection the
      piece is straight;
    * exact for data sampled from any constant, linear or quadratic function
      that is monotone on each interval (then c_L = c_R = f'' and the cap is
      slack, so (*) reproduces the true average);
    * local: f on I_k depends only on f_{k-1}, ..., f_{k+2};
    * FLAT extrapolation: f_0 before the first node, f_{n-1} after the last.
      With one node the curve is that constant (required for the funding curve).

    Value / gradient consistency
    ----------------------------
    Every quantity above is LINEAR in the node vector once the regime (the
    sign test in step 2 and the cap in step 3) is fixed by the data.  The
    class therefore carries each scalar as a "dual" row [value, d/df_0, ...,
    d/df_{n-1}] through one code path, so ``interpolate`` /
    ``gradient_wrt_ordinate`` (and the integral pair) are two columns of the
    same computation and cannot disagree.  As with PCP's ``searchsorted``
    tie-break, the gradient is the one-sided derivative exactly at a regime
    boundary.
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values, InterpMethod.MONOTONE_CONVEX,
                         extrapolation_method)
        if self.extrap_method_ != ExtrapMethod.FLAT:
            raise ValueError("Interpolator1DMonotoneConvex supports FLAT extrapolation only")

        x = np.asarray(self.axis1_, dtype=float)
        f = np.asarray(self.values_, dtype=float)
        n = len(x)
        if n == 0:
            raise ValueError("at least one node is required")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(f))):
            raise ValueError("node times and values must be finite")
        h = np.diff(x)
        if np.any(h <= 0.0):
            raise ValueError("node times must be strictly increasing")

        self.x_ = x
        self.h_ = h
        # Dual node values: column 0 is the value, columns 1.. the gradient
        # w.r.t. the ordinate vector (the identity, since f_j is itself a node).
        self.F_ = np.hstack([f[:, None], np.eye(n)])
        if n >= 2:
            self.M_, self.c_ = self._interval_averages(self.F_, x, h)   # (n-1, n+1), (n-1,)
            self.G0_ = self.F_[:-1] - self.M_                      # g0_k = f_k     - m_k
            self.G1_ = self.F_[1:] - self.M_                       # g1_k = f_{k+1} - m_k
        else:
            self.M_ = self.G0_ = self.G1_ = np.zeros((0, n + 1))
            self.c_ = np.zeros(0)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @staticmethod
    def _interval_averages(F: np.ndarray, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Dual interval averages m_k (n-1 rows) and the curvatures c_k used.

        Linear algebra on dual rows is ordinary numpy arithmetic; the only
        data-dependent decisions are the sign test and the cap, taken on the
        value column and then applied as constant multipliers, so the gradient
        columns are exact within the regime.
        """
        n = F.shape[0]
        delta = (F[1:] - F[:-1]) / h[:, None]                      # secant slopes, (n-1, n+1)
        mid = 0.5 * (F[:-1] + F[1:])                               # chord midpoints
        if n == 2:
            return mid, np.zeros(1)                                # one interval: straight line

        # Second derivative of the parabola through nodes j-1, j, j+1,
        # attached to the interior node j = 1..n-2:  2 * f[t_{j-1}, t_j, t_{j+1}].
        c_node = 2.0 * (delta[1:] - delta[:-1]) / (x[2:] - x[:-2])[:, None]   # (n-2, n+1)

        c = np.zeros_like(delta)
        for k in range(n - 1):
            has_left, has_right = k >= 1, k <= n - 3
            if has_left and has_right:
                cl, cr = c_node[k - 1], c_node[k]
                if cl[0] * cr[0] < 0.0:
                    continue                                       # data inflects: straight piece
                c[k] = 0.5 * (cl + cr)
            elif has_left:
                c[k] = c_node[k - 1]
            else:
                c[k] = c_node[k]
            # Monotonicity cap |c_k| <= 2 |delta_k| / h_k, applied as a rescale
            # (a constant factor in the regime) so the dual row stays consistent.
            limit = 2.0 * abs(delta[k, 0]) / h[k]
            if abs(c[k, 0]) > limit:
                c[k] = c[k] * (limit / abs(c[k, 0])) if limit > 0.0 else 0.0 * c[k]
        return mid - (h ** 2 / 12.0)[:, None] * c, np.array(c[:, 0], dtype=float)

    # ------------------------------------------------------------------
    # Dual evaluation: one code path for value and gradient
    # ------------------------------------------------------------------
    def _piece(self, x: float) -> int:
        """Index k of the piece [t_k, t_{k+1}] containing x (interior x only)."""
        return min(max(int(np.searchsorted(self.x_, x, side='right')) - 1, 0), self.length_ - 2)

    def _value_dual(self, x: float) -> np.ndarray:
        if self.length_ == 1 or x <= self.x_[0]:
            return self.F_[0]                                      # left FLAT wing
        if x >= self.x_[-1]:
            return self.F_[-1]                                     # right FLAT wing
        k = self._piece(x)
        u = (x - self.x_[k]) / self.h_[k]
        return self.M_[k] + self.G0_[k] * (1.0 - 4.0 * u + 3.0 * u * u) + self.G1_[k] * (-2.0 * u + 3.0 * u * u)

    def _piece_antiderivative(self, k: int, u: float) -> np.ndarray:
        """Dual value of (1/h_k) * int_{t_k}^{t_k + u h_k} f(s) ds, i.e. in u-units.

        Integrating (HW 47) term by term:
            int_0^u (1 - 4u + 3u^2) = u - 2u^2 + u^3 = u (1 - u)^2
            int_0^u (-2u + 3u^2)    = -u^2 + u^3    = u^2 (u - 1)
        At u = 1 both vanish, leaving m_k: the piece integrates to h_k m_k.
        """
        return self.M_[k] * u + self.G0_[k] * (u * (1.0 - u) ** 2) + self.G1_[k] * (u * u * (u - 1.0))

    def _integral_dual(self, a: float, b: float) -> np.ndarray:
        if a > b:
            return -self._integral_dual(b, a)
        total = np.zeros(self.length_ + 1)
        if self.length_ == 1:
            return total + (b - a) * self.F_[0]
        x0, xn = self.x_[0], self.x_[-1]
        # Left wing: constant f_0 on (-inf, t_0].
        if a < x0:
            total += (min(b, x0) - a) * self.F_[0]
        # Interior pieces that overlap [a, b].
        if b > x0 and a < xn:
            lo_k = self._piece(max(a, x0))
            hi_k = self._piece(min(b, xn)) if min(b, xn) < xn else self.length_ - 2
            for k in range(lo_k, hi_k + 1):
                lo = max(a, self.x_[k])
                hi = min(b, self.x_[k + 1])
                if hi > lo:
                    u_lo = (lo - self.x_[k]) / self.h_[k]
                    u_hi = (hi - self.x_[k]) / self.h_[k]
                    total += self.h_[k] * (self._piece_antiderivative(k, u_hi) - self._piece_antiderivative(k, u_lo))
        # Right wing: constant f_{n-1} on [t_{n-1}, inf).
        if b > xn:
            total += (b - max(a, xn)) * self.F_[-1]
        return total

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def interpolate(self, x: float) -> float:
        return float(self._value_dual(x)[0])

    def integrate(self, start_x: float, end_x: float) -> float:
        return float(self._integral_dual(start_x, end_x)[0])

    # Optional risk interface: the gradient columns of the same dual rows.
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        return np.array(self._value_dual(x)[1:], dtype=float)

    def gradient_of_integrated_value_wrt_ordinate(
        self, start_x: float, end_x: float,
    ) -> np.ndarray:
        return np.array(self._integral_dual(start_x, end_x)[1:], dtype=float)

    # Diagnostics used by the notebook plot and the tests.
    @property
    def interval_averages(self) -> np.ndarray:
        """m_k: the average IFR on each node interval, so D(t_{k+1})/D(t_k) = exp(-h_k m_k)."""
        return np.array(self.M_[:, 0], dtype=float)

    @property
    def interval_curvatures(self) -> np.ndarray:
        """c_k = f'' on each node interval (constant per piece), as chosen in step 3."""
        return np.array(self.c_, dtype=float)


class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        elif interpolation_method == InterpMethod.MONOTONE_CONVEX:
            return Interpolator1DMonotoneConvex(axis1_, values_, extrapolation_method)
        else:
            raise ValueError(f'Unsupported interpolation method: {interpolation_method}')
