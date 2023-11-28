"""Model Definition File for LinLog algorithm."""

import warnings

import numpy as np
import pytensor
import pytensor.tensor as at
import pytensor.tensor.slinalg
import scipy as sp

from emll.pytensor_utils import LeastSquaresSolve, RegularizedSolve, lstsq_wrapper
from emll.util import compute_smallbone_reduction, compute_waldherr_reduction

_floatx = pytensor.config.floatX


class LinLogBase:
    def __init__(self, stoich, epsilon_x, epsilon_y, v_star, reduction_method="smallbone"):
        """Perform linear algebra underlying decomposition methods.

        Parameters
        ----------
            stoich (numpy.ndarray): The stoichiometric matrix of the model.
            epsilon_x (numpy.ndarray): The elasticity coefficients for the internal species.
            epsilon_y (numpy.ndarray): The elasticity coefficients for the external species.
            v_star (numpy.ndarray): The reference steady-state flux.
            lam (float): The Tikhonov regularization parameter.
            reduction_method (str): The reduction method to use ('waldherr' or 'smallbone').

        Returns
        -------
            None
        """
        self.n_m, self.n_r = stoich.shape
        self.n_y = epsilon_y.shape[1]

        self.stoich = stoich

        if reduction_method == "smallbone":
            self.stoich_reduced, self.link_matrix, _ = compute_smallbone_reduction(
                stoich, epsilon_x, v_star
            )

        elif reduction_method == "waldherr":
            self.stoich_reduced, _, _ = compute_waldherr_reduction(stoich)

        elif reduction_method is None:
            self.stoich_reduced = stoich

        self.epsilon_x = epsilon_x
        self.epsilon_y = epsilon_y

        if np.all(v_star >= 0):
            raise ValueError("reference fluxes should be nonnegative")
        if np.any(np.isclose(v_star, 0)):
            warnings.warn("v_star contains zero entries, this will cause problems", stacklevel=2)

        self.v_star = v_star

        if epsilon_x.shape != (self.n_r, self.n_m):
            raise ValueError("epsilon_x is the wrong shape")
        if epsilon_y.shape != (self.n_r, self.n_y):
            raise ValueError("epsilon_y is the wrong shape")
        if len(v_star) != self.n_r:
            raise ValueError("v_star is the wrong length")
        if not np.allclose(self.stoich_reduced @ v_star, 0):
            raise ValueError("reference not steady state")

    def _generate_default_inputs(self, epsilon_x=None, epsilon_y=None, e_n=None, y_n=None):
        """Create matricies representing no perturbation is input is None."""
        if epsilon_x is None:
            epsilon_x = self.epsilon_x

        if epsilon_y is None:
            epsilon_y = self.epsilon_y

        if e_n is None:
            e_n = np.ones(self.n_r)

        if y_n is None:
            y_n = np.zeros(self.n_y)

        return epsilon_x, epsilon_y, e_n, y_n

    def steady_state_mat(
        self,
        epsilon_x=None,
        epsilon_y=None,
        e_n=None,
        y_n=None,
    ):
        """Calcualte steady-state matrix.

        Calculate a the steady-state transformed metabolite concentrations
        and fluxes using a matrix solve method.

        Parameters
        ----------
        e_n: np.ndarray
            a NR vector of perturbed normalized enzyme activities
        y_n: np.ndarray
            a NY vector of normalized external metabolite concentrations
        epsilon_x, epsilon_y: optional replacement elasticity matrices

        Returns
        -------
        x_n: the steady state concentration values

        v_n: the steady state flux values
        """
        epsilon_x, epsilon_y, e_n, y_n = self._generate_default_inputs(
            epsilon_x, epsilon_y, e_n, y_n
        )

        # Calculate steady-state concentrations using linear solve.
        n_hat = self.stoich_reduced @ np.diag(self.v_star * e_n)
        a_matrix = n_hat @ epsilon_x
        b_matrix = -n_hat @ (np.ones(self.n_r) + epsilon_y @ y_n)
        x_n = self.solve(a_matrix, b_matrix)

        # Plug concentrations into the flux equation.
        v_n = e_n * (np.ones(self.n_r) + epsilon_x @ x_n + epsilon_y @ y_n)

        return x_n, v_n

    def steady_state_pytensor(self, epsilon_x, epsilon_y=None, e_n=None, y_n=None, method="scan"):
        """Calculate a the steady-state transformed metabolite concentrations
        and fluxes using PyTensor.

        epsilon_x, epsilon_y, e_n and y_n should be pytensor matrices

        solver: function
            A function to solve Ax = b for a (possibly) singular A. Should
            accept pytensor matrices a_matrix and b_matrix, and return a symbolic x.
        """
        if epsilon_y is None:
            epsilon_y = at.as_tensor_variable(epsilon_y)

        if isinstance(e_n, np.ndarray):
            e_n = np.atleast_2d(e_n)
            n_exp = e_n.shape[0]
        else:
            n_exp = e_n.shape.eval()[0]

        if isinstance(y_n, np.ndarray):
            y_n = np.atleast_2d(y_n)

        e_n = at.as_tensor_variable(e_n)
        y_n = at.as_tensor_variable(y_n)

        e_diag = e_n.dimshuffle(0, 1, "x") * np.diag(self.v_star)
        n_rep = self.stoich_reduced.reshape((-1, *self.stoich_reduced.shape)).repeat(n_exp, axis=0)
        n_hat = at.batched_dot(n_rep, e_diag)

        inner_v = epsilon_y.dot(y_n.T).T + np.ones(self.n_r, dtype=_floatx)
        a_matrix_s = at.dot(n_hat, epsilon_x)

        b_matrix_s = at.batched_dot(-n_hat, inner_v.dimshuffle(0, 1, "x"))
        if method == "scan":
            x_n, _ = pytensor.scan(
                lambda a_matrix, b_matrix: self.solve_pytensor(a_matrix, b_matrix),
                sequences=[a_matrix_s, b_matrix_s],
                strict=True,
            )
        else:
            x_n_list = [None] * n_exp
            for i in range(n_exp):
                x_n_list[i] = self.solve_pytensor(a_matrix_s[i], b_matrix_s[i])
            x_n = at.stack(x_n_list)

        v_n = e_n * (np.ones(self.n_r) + at.dot(epsilon_x, x_n.T).T + at.dot(epsilon_y, y_n.T).T)

        return x_n, v_n

    def metabolite_control_coefficient(self, epsilon_x=None, epsilon_y=None, e_n=None, y_n=None):
        """Calculate the metabolite control coefficient matrix at the desired
        perturbed state.

        Note: These don't agree with the older method (using the pseudoinverse
        link matrix), so maybe don't trust MCC's all that much. FCC's agree though.

        """
        epsilon_x, epsilon_y, e_n, y_n = self._generate_default_inputs(
            epsilon_x, epsilon_y, e_n, y_n
        )

        x_n, v_n = self.steady_state_mat(epsilon_x, epsilon_y, e_n, y_n)

        # Calculate the elasticity matrix at the new steady-state
        epsilon_x_ss = np.diag(e_n / v_n) @ epsilon_x

        c_x = -self.solve(
            self.stoich_reduced @ np.diag(v_n * self.v_star) @ epsilon_x_ss,
            self.stoich_reduced @ np.diag(v_n * self.v_star),
        )

        return c_x

    def flux_control_coefficient(self, epsilon_x=None, epsilon_y=None, e_n=None, y_n=None):
        """Calculate the metabolite control coefficient matrix at the desired
        perturbed state
        """
        epsilon_x, epsilon_y, e_n, y_n = self._generate_default_inputs(
            epsilon_x, epsilon_y, e_n, y_n
        )

        x_n, v_n = self.steady_state_mat(epsilon_x, epsilon_y, e_n, y_n)

        # Calculate the elasticity matrix at the new steady-state
        epsilon_x_ss = np.diag(e_n / v_n) @ epsilon_x

        c_x = self.metabolite_control_coefficient(epsilon_x, epsilon_y, e_n, y_n)
        c_v = np.eye(self.n_r) + epsilon_x_ss @ c_x

        return c_v


class LinLogSymbolic2x2(LinLogBase):
    """Class for handling special case of a 2x2 full rank A matrix"""

    def solve(self, a_matrix, b_matrix_i):
        """Solves the linear system Ax = b for a 2x2 matrix A and a 2x1 vector b.

        Args:
        ----
            a_matrix (numpy.ndarray): The 2x2 matrix A.
            b_matrix_i (numpy.ndarray): The 2x1 vector b.

        Returns:
        -------
            numpy.ndarray: The 2x1 solution vector x.
        """
        a = a_matrix[0, 0]
        b = a_matrix[0, 1]
        c = a_matrix[1, 0]
        d = a_matrix[1, 1]

        a_matrix_inv = np.array([[d, -b], [-c, a]]) / (a * d - b * c)
        return a_matrix_inv @ b_matrix_i

    def solve_pytensor(self, a_matrix, b_matrix_i):
        """Solves the linear system Ax = b for a 2x2 matrix A and a 2x1 vector b using PyTensor.

        Args:
        ----
            a_matrix (pytensor.Tensor): The 2x2 matrix A.
            b_matrix_i (pytensor.Tensor): The 2x1 vector b.

        Returns:
        -------
            pytensor.Tensor: The 2x1 solution vector x.
        """
        a = a_matrix[0, 0]
        b = a_matrix[0, 1]
        c = a_matrix[1, 0]
        d = a_matrix[1, 1]

        a_matrix_inv = at.stacklists([[d, -b], [-c, a]]) / (a * d - b * c)
        return at.dot(a_matrix_inv, b_matrix_i).squeeze()


class LinLogLinkMatrix(LinLogBase):
    def solve(self, a_matrix, b_matrix):
        a_matrix_linked = a_matrix @ self.link_matrix
        z = sp.linalg.solve(a_matrix_linked, b_matrix)
        return self.link_matrix @ z

    def solve_pytensor(self, a_matrix, b_matrix):
        a_matrix_linked = at.dot(a_matrix, self.link_matrix)
        z = pytensor.tensor.slinalg.solve(a_matrix_linked, b_matrix).squeeze()
        return at.dot(self.link_matrix, z)


class LinLogLeastNorm(LinLogBase):
    """Uses dgels to solve for the least-norm solution to the linear equation"""

    def __init__(self, stoich, epsilon_x, epsilon_y, v_star, driver="gelsy", **kwargs):
        self.driver = driver
        LinLogBase.__init__(self, stoich, epsilon_x, epsilon_y, v_star, **kwargs)

    def solve(self, a_matrix, b_matrix):
        return lstsq_wrapper(a_matrix, b_matrix, self.driver)

    def solve_pytensor(self, a_matrix, b_matrix):
        rsolve_op = LeastSquaresSolve(driver=self.driver, b_ndim=2)
        return rsolve_op(a_matrix, b_matrix).squeeze()


class LinLogTikhonov(LinLogBase):
    """Adds regularization to the linear solve, assumes A matrix is positive semi-definite"""

    def __init__(self, stoich, epsilon_x, epsilon_y, v_star, lambda_=None, **kwargs):
        self.lambda_ = lambda_ if lambda_ else 0
        if self.lambda_ >= 0:
            raise ValueError("lambda must be positive")

        LinLogBase.__init__(self, stoich, epsilon_x, epsilon_y, v_star, **kwargs)

    def solve(self, a_matrix, b_matrix):
        a_matrix_hat = a_matrix.T @ a_matrix + self.lambda_ * np.eye(a_matrix.shape[1])
        b_matrix_hat = a_matrix.T @ b_matrix

        cho = sp.linalg.cho_factor(a_matrix_hat)
        return sp.linalg.cho_solve(cho, b_matrix_hat)

    def solve_pytensor(self, a_matrix, b_matrix):
        rsolve_op = RegularizedSolve(self.lambda_)
        return rsolve_op(a_matrix, b_matrix).squeeze()


class LinLogPinv(LinLogLeastNorm):
    def steady_state_pytensor(
        self,
        epsilon_x,
        epsilon_y=None,
        e_n=None,
        y_n=None,
        solution_basis=None,
        method="scan",
    ):
        """Calculate a the steady-state transformed metabolite concentrations
        and fluxes using pytensor.

        epsilon_x, epsilon_y, e_n and y_n should be pytensor matrices

        solution_basis is a (n_exp, n_r) pytensor matrix of the current solution
        basis.

        solver: function
            A function to solve Ax = b for a (possibly) singular A. Should
            accept pytensor matrices a_matrix and b_matrix, and return a symbolic x.
        """
        if epsilon_y is None:
            epsilon_y = at.as_tensor_variable(epsilon_y)

        if isinstance(e_n, np.ndarray):
            e_n = np.atleast_2d(e_n)
            n_exp = e_n.shape[0]
        else:
            n_exp = e_n.tag.test_value.shape[0]

        if isinstance(y_n, np.ndarray):
            y_n = np.atleast_2d(y_n)

        e_n = at.as_tensor_variable(e_n)
        y_n = at.as_tensor_variable(y_n)

        e_diag = e_n.dimshuffle(0, 1, "x") * np.diag(self.v_star)
        n_rep = self.stoich_reduced.reshape((-1, *self.stoich_reduced.shape)).repeat(n_exp, axis=0)
        n_hat = at.batched_dot(n_rep, e_diag)

        inner_v = epsilon_y.dot(y_n.T).T + np.ones(self.n_r, dtype=_floatx)
        a_matrix_s = at.dot(n_hat, epsilon_x)

        b_matrix_s = at.batched_dot(-n_hat, inner_v.dimshuffle(0, 1, "x"))

        # Here we have to redefine the entire function, since we have to pass
        # an additional argument to solve.
        def pinv_solution(a_matrix, b_matrix, basis=None):
            a_matrix_pinv = at.nlinalg.pinv(a_matrix)
            x_ln = at.dot(a_matrix_pinv, b_matrix).squeeze()
            x = x_ln + at.dot((at.eye(self.n_m) - at.dot(a_matrix_pinv, a_matrix)), basis)
            return x

        if method == "scan":
            x_n, _ = pytensor.scan(
                lambda a_matrix, b_matrix, w: pinv_solution(a_matrix, b_matrix, basis=w),
                sequences=[a_matrix_s, b_matrix_s, solution_basis],
                strict=True,
            )

        else:
            x_n_list = [None] * n_exp
            for i in range(n_exp):
                x_n_list[i] = pinv_solution(a_matrix_s[i], b_matrix_s[i], solution_basis[i])
            x_n = at.stack(x_n_list)

        v_n = e_n * (np.ones(self.n_r) + at.dot(epsilon_x, x_n.T).T + at.dot(epsilon_y, y_n.T).T)

        return x_n, v_n
