""" Model Definition File for LinLog algorithm """
import warnings

import numpy as np
import scipy as sp

import pytensor
import pytensor.tensor as at
import pytensor.tensor.slinalg

from emll.pytensor_utils import RegularizedSolve, LeastSquaresSolve, lstsq_wrapper
from emll.util import compute_smallbone_reduction, compute_waldherr_reduction


_floatX = pytensor.config.floatX


class LinLogBase:
    def __init__(self, N, Ex, Ey, v_star, reduction_method="smallbone"):
        """A class to perform the linear algebra underlying the
        decomposition method.


        Parameters
        ----------
        N : np.array
            The full stoichiometric matrix of the considered model. Must be of
            dimensions MxN
        Ex : np.array
            An NxM array of the elasticity coefficients for the given linlog
            model.
        Ey : np.array
            An NxP array of the elasticity coefficients for the external
            species.
        v_star : np.array
            A length M vector specifying the original steady-state flux
            solution of the model.
        lam : float
            The λ value to use for tikhonov regularization
        reduction_method : 'waldherr', 'smallbone', or None
            Type of stoichiometric decomposition to perform (default
            'smallbone')


        """
        self.nm, self.nr = N.shape
        self.ny = Ey.shape[1]

        self.N = N

        if reduction_method == "smallbone":
            self.Nr, self.L, _ = compute_smallbone_reduction(N, Ex, v_star)

        elif reduction_method == "waldherr":
            self.Nr, _, _ = compute_waldherr_reduction(N)

        elif reduction_method is None:
            self.Nr = N

        self.Ex = Ex
        self.Ey = Ey

        assert np.all(v_star >= 0), "reference fluxes should be nonnegative"
        if np.any(np.isclose(v_star, 0)):
            warnings.warn("v_star contains zero entries, this will cause problems")

        self.v_star = v_star

        assert Ex.shape == (self.nr, self.nm), "Ex is the wrong shape"
        assert Ey.shape == (self.nr, self.ny), "Ey is the wrong shape"
        assert len(v_star) == self.nr, "v_star is the wrong length"
        assert np.allclose(self.Nr @ v_star, 0), "reference not steady state"

    def _generate_default_inputs(self, Ex=None, Ey=None, en=None, yn=None):
        """Create matricies representing no perturbation is input is None."""
        if Ex is None:
            Ex = self.Ex

        if Ey is None:
            Ey = self.Ey

        if en is None:
            en = np.ones(self.nr)

        if yn is None:
            yn = np.zeros(self.ny)

        return Ex, Ey, en, yn

    def steady_state_mat(
        self,
        Ex=None,
        Ey=None,
        en=None,
        yn=None,
    ):
        """Calculate a the steady-state transformed metabolite concentrations
        and fluxes using a matrix solve method.

        en: np.ndarray
            a NR vector of perturbed normalized enzyme activities
        yn: np.ndarray
            a NY vector of normalized external metabolite concentrations
        Ex, Ey: optional replacement elasticity matrices

        """
        Ex, Ey, en, yn = self._generate_default_inputs(Ex, Ey, en, yn)

        # Calculate steady-state concentrations using linear solve.
        N_hat = self.Nr @ np.diag(self.v_star * en)
        A = N_hat @ Ex
        b = -N_hat @ (np.ones(self.nr) + Ey @ yn)
        xn = self.solve(A, b)

        # Plug concentrations into the flux equation.
        vn = en * (np.ones(self.nr) + Ex @ xn + Ey @ yn)

        return xn, vn

    def steady_state_pytensor(self, Ex, Ey=None, en=None, yn=None, method="scan"):
        """Calculate a the steady-state transformed metabolite concentrations
        and fluxes using PyTensor.

        Ex, Ey, en and yn should be pytensor matrices

        solver: function
            A function to solve Ax = b for a (possibly) singular A. Should
            accept pytensor matrices A and b, and return a symbolic x.
        """

        if Ey is None:
            Ey = at.as_tensor_variable(Ey)

        if isinstance(en, np.ndarray):
            en = np.atleast_2d(en)
            n_exp = en.shape[0]
        else:
            n_exp = en.eval().shape[0]

        if isinstance(yn, np.ndarray):
            yn = np.atleast_2d(yn)

        en = at.as_tensor_variable(en)
        yn = at.as_tensor_variable(yn)

        e_diag = en.dimshuffle(0, 1, "x") * np.diag(self.v_star)
        N_rep = self.Nr.reshape((-1, *self.Nr.shape)).repeat(n_exp, axis=0)
        N_hat = at.batched_dot(N_rep, e_diag)

        inner_v = Ey.dot(yn.T).T + np.ones(self.nr, dtype=_floatX)
        As = at.dot(N_hat, Ex)

        bs = at.batched_dot(-N_hat, inner_v.dimshuffle(0, 1, "x"))
        if method == "scan":
            xn, _ = pytensor.scan(
                lambda A, b: self.solve_pytensor(A, b), sequences=[As, bs], strict=True
            )
        else:
            xn_list = [None] * n_exp
            for i in range(n_exp):
                xn_list[i] = self.solve_pytensor(As[i], bs[i])
            xn = at.stack(xn_list)

        vn = en * (np.ones(self.nr) + at.dot(Ex, xn.T).T + at.dot(Ey, yn.T).T)

        return xn, vn

    def metabolite_control_coefficient(self, Ex=None, Ey=None, en=None, yn=None):
        """Calculate the metabolite control coefficient matrix at the desired
        perturbed state.

        Note: These don't agree with the older method (using the pseudoinverse
        link matrix), so maybe don't trust MCC's all that much. FCC's agree though.

        """

        Ex, Ey, en, yn = self._generate_default_inputs(Ex, Ey, en, yn)

        xn, vn = self.steady_state_mat(Ex, Ey, en, yn)

        # Calculate the elasticity matrix at the new steady-state
        Ex_ss = np.diag(en / vn) @ Ex

        Cx = -self.solve(
            self.Nr @ np.diag(vn * self.v_star) @ Ex_ss,
            self.Nr @ np.diag(vn * self.v_star),
        )

        return Cx

    def flux_control_coefficient(self, Ex=None, Ey=None, en=None, yn=None):
        """Calculate the metabolite control coefficient matrix at the desired
        perturbed state"""

        Ex, Ey, en, yn = self._generate_default_inputs(Ex, Ey, en, yn)

        xn, vn = self.steady_state_mat(Ex, Ey, en, yn)

        # Calculate the elasticity matrix at the new steady-state
        Ex_ss = np.diag(en / vn) @ Ex

        Cx = self.metabolite_control_coefficient(Ex, Ey, en, yn)
        Cv = np.eye(self.nr) + Ex_ss @ Cx

        return Cv


class LinLogSymbolic2x2(LinLogBase):
    """Class for handling special case of a 2x2 full rank A matrix"""

    def solve(self, A, bi):
        a = A[0, 0]
        b = A[0, 1]
        c = A[1, 0]
        d = A[1, 1]

        A_inv = np.array([[d, -b], [-c, a]]) / (a * d - b * c)
        return A_inv @ bi

    def solve_pytensor(self, A, bi):
        a = A[0, 0]
        b = A[0, 1]
        c = A[1, 0]
        d = A[1, 1]

        A_inv = at.stacklists([[d, -b], [-c, a]]) / (a * d - b * c)
        return at.dot(A_inv, bi).squeeze()


class LinLogLinkMatrix(LinLogBase):
    def solve(self, A, b):
        A_linked = A @ self.L
        z = sp.linalg.solve(A_linked, b)
        return self.L @ z

    def solve_pytensor(self, A, b):
        A_linked = at.dot(A, self.L)
        z = pytensor.tensor.slinalg.solve(A_linked, b).squeeze()
        return at.dot(self.L, z)


class LinLogLeastNorm(LinLogBase):
    """Uses dgels to solve for the least-norm solution to the linear equation"""

    def __init__(self, N, Ex, Ey, v_star, driver="gelsy", **kwargs):
        self.driver = driver
        LinLogBase.__init__(self, N, Ex, Ey, v_star, **kwargs)

    def solve(self, A, b):
        return lstsq_wrapper(A, b, self.driver)

    def solve_pytensor(self, A, b):
        rsolve_op = LeastSquaresSolve(driver=self.driver, b_ndim=2)
        return rsolve_op(A, b).squeeze()


class LinLogTikhonov(LinLogBase):
    """Adds regularization to the linear solve, assumes A matrix is positive semi-definite"""

    def __init__(self, N, Ex, Ey, v_star, lambda_=None, **kwargs):
        self.lambda_ = lambda_ if lambda_ else 0
        assert self.lambda_ >= 0, "lambda must be positive"

        LinLogBase.__init__(self, N, Ex, Ey, v_star, **kwargs)

    def solve(self, A, b):
        A_hat = A.T @ A + self.lambda_ * np.eye(A.shape[1])
        b_hat = A.T @ b

        cho = sp.linalg.cho_factor(A_hat)
        return sp.linalg.cho_solve(cho, b_hat)

    def solve_pytensor(self, A, b):
        rsolve_op = RegularizedSolve(self.lambda_)
        return rsolve_op(A, b).squeeze()


class LinLogPinv(LinLogLeastNorm):
    def steady_state_pytensor(
        self,
        Ex,
        Ey=None,
        en=None,
        yn=None,
        solution_basis=None,
        method="scan",
        driver="gelsy",
    ):
        """Calculate a the steady-state transformed metabolite concentrations
        and fluxes using pytensor.

        Ex, Ey, en and yn should be pytensor matrices

        solution_basis is a (n_exp, nr) pytensor matrix of the current solution
        basis.

        solver: function
            A function to solve Ax = b for a (possibly) singular A. Should
            accept pytensor matrices A and b, and return a symbolic x.
        """

        if Ey is None:
            Ey = at.as_tensor_variable(Ey)

        if isinstance(en, np.ndarray):
            en = np.atleast_2d(en)
            n_exp = en.shape[0]
        else:
            n_exp = en.tag.test_value.shape[0]

        if isinstance(yn, np.ndarray):
            yn = np.atleast_2d(yn)

        en = at.as_tensor_variable(en)
        yn = at.as_tensor_variable(yn)

        e_diag = en.dimshuffle(0, 1, "x") * np.diag(self.v_star)
        N_rep = self.Nr.reshape((-1, *self.Nr.shape)).repeat(n_exp, axis=0)
        N_hat = at.batched_dot(N_rep, e_diag)

        inner_v = Ey.dot(yn.T).T + np.ones(self.nr, dtype=_floatX)
        As = at.dot(N_hat, Ex)

        bs = at.batched_dot(-N_hat, inner_v.dimshuffle(0, 1, "x"))

        # Here we have to redefine the entire function, since we have to pass
        # an additional argument to solve.
        def pinv_solution(A, b, basis=None):
            A_pinv = at.nlinalg.pinv(A)
            x_ln = at.dot(A_pinv, b).squeeze()
            x = x_ln + at.dot((at.eye(self.nm) - at.dot(A_pinv, A)), basis)
            return x

        if method == "scan":
            xn, _ = pytensor.scan(
                lambda A, b, w: pinv_solution(A, b, basis=w),
                sequences=[As, bs, solution_basis],
                strict=True,
            )

        else:
            xn_list = [None] * n_exp
            for i in range(n_exp):
                xn_list[i] = pinv_solution(As[i], bs[i], solution_basis[i])
            xn = at.stack(xn_list)

        vn = en * (np.ones(self.nr) + at.dot(Ex, xn.T).T + at.dot(Ey, yn.T).T)

        return xn, vn
