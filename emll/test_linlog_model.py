""" Test File for linlog model methods """
import numpy as np
import pytest

import pytensor
import pytensor.tensor as at

from emll.test_models import models
from emll.linlog_model import LinLogLeastNorm, LinLogLinkMatrix, LinLogTikhonov
from emll.util import create_elasticity_matrix, create_Ey_matrix

pytensor.config.compute_test_value = "ignore"
pytensor.config.optimizer = "fast_compile"


@pytest.fixture(
    params=["teusink", "mendes", "textbook", "greene_small", "greene_large", "contador"],
    name="cobra_model"
)
def cobra_model_fixture(request):
    '''Loads in cobra model and sets elasticity'''
    model, N, v_star = models[request.param]()
    Ex = create_elasticity_matrix(model)
    Ey = create_Ey_matrix(model)
    return model, N, Ex, Ey, v_star


@pytest.fixture(name="linlog_least_norm")
def linlog_least_norm_fixture(cobra_model):
    '''Method for calculating linlog least norm'''
    _, N, Ex, Ey, v_star = cobra_model
    return LinLogLeastNorm(N, Ex, Ey, v_star)


@pytest.fixture()
def linlog_tikhonov_fixture(cobra_model):
    _, N, Ex, Ey, v_star = cobra_model
    return LinLogTikhonov(N, Ex, Ey, v_star, lambda_=1e-6)


@pytest.fixture()
def linlog_link_fixture(cobra_model):
    _, N, Ex, Ey, v_star = cobra_model
    return LinLogLinkMatrix(N, Ex, Ey, v_star)


@pytest.fixture(
    params=[linlog_least_norm_fixture, linlog_tikhonov_fixture, linlog_link_fixture],
    name="linlog_model"
)
def linlog_model_fixture(request, cobra_model):
    fixture = request.param
    return fixture(cobra_model)


def test_steady_state(linlog_model):
    '''Test to ensure steady state'''
    ll = linlog_model

    # Fake up some experiments
    n_exp = 1

    e_hat_np = 2 ** (0.5 * np.random.randn(n_exp, ll.nr))
    y_hat_np = 2 ** (0.5 * np.random.randn(n_exp, ll.ny))

    e_hat_t = at.dmatrix("en")
    e_hat_t.tag.test_value = e_hat_np

    y_hat_t = at.dmatrix("yn")
    y_hat_t.tag.test_value = y_hat_np

    Ex_t = at.dmatrix("Ex")
    Ex_t.tag.test_value = ll.Ex

    Ey_t = at.dmatrix("Ey")
    Ey_t.tag.test_value = ll.Ey

    chi_ss, v_hat_ss = ll.steady_state_pytensor(Ex_t, Ey_t, e_hat_t, y_hat_t)

    io_fun = pytensor.function([Ex_t, Ey_t, e_hat_t, y_hat_t], [chi_ss, v_hat_ss])
    x_pytensor_test, v_pytensor_test = io_fun(ll.Ex, ll.Ey, e_hat_np, y_hat_np)

    x_np, v_np = ll.steady_state_mat(ll.Ex, ll.Ey, e_hat_np[0], y_hat_np[0])

    np.testing.assert_allclose(x_np, x_pytensor_test.flatten(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(v_np, v_pytensor_test.flatten(), atol=1e-5, rtol=1e-5)


def test_control_coeff(linlog_model):
    ''' Test for control coeffecients'''
    ll = linlog_model

    fd = 1e-5
    es = np.ones((ll.nr, ll.nr)) + fd * np.eye(ll.nr)

    cx_fd = np.zeros((ll.nm, ll.nr))
    cv_fd = np.zeros((ll.nr, ll.nr))
    for i, ei in enumerate(es):
        xni, vni = ll.steady_state_mat(en=ei)
        cx_fd[:, i] = xni / fd
        cv_fd[:, i] = (vni - 1) / fd

    np.testing.assert_allclose(
        cx_fd, ll.metabolite_control_coefficient(en=ei), atol=1e-5, rtol=1e-4
    )
    np.testing.assert_allclose(
        cv_fd, ll.flux_control_coefficient(en=ei), atol=1e-5, rtol=1e-4
    )


def test_reduction_methods(cobra_model):
    '''Test for reduction methods'''
    model, N, Ex, Ey, v_star = cobra_model
    ll1 = LinLogLeastNorm(N, Ex, Ey, v_star, reduction_method="smallbone")
    ll2 = LinLogLeastNorm(N, Ex, Ey, v_star, reduction_method="waldherr")
    ll3 = LinLogLeastNorm(N, Ex, Ey, v_star, reduction_method=None)

    n_exp = 1
    e_hat_np = 2 ** (0.5 * np.random.randn(n_exp, ll1.nr))
    y_hat_np = 2 ** (0.5 * np.random.randn(n_exp, ll1.ny))

    x1, v1 = ll1.steady_state_mat(en=e_hat_np[0], yn=y_hat_np[0])
    x2, v2 = ll2.steady_state_mat(en=e_hat_np[0], yn=y_hat_np[0])
    x3, v3 = ll3.steady_state_mat(en=e_hat_np[0], yn=y_hat_np[0])

    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(x2, x3)
    np.testing.assert_allclose(v1, v2)
    np.testing.assert_allclose(v2, v3)

    np.testing.assert_allclose(ll1.Nr @ (v1 * v_star), 0.0, atol=1e-10)
    np.testing.assert_allclose(ll2.Nr @ (v2 * v_star), 0.0, atol=1e-10)
    np.testing.assert_allclose(ll3.Nr @ (v3 * v_star), 0.0, atol=1e-10)
