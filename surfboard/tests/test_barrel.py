import pytest

import numpy as np

from surfboard import statistics


@pytest.fixture
def barrel():
    feature = np.ones((25, 267))
    return statistics.Barrel(feature)


def test_barrel_constructor(barrel):
    feat = barrel()
    assert (feat == np.ones((25, 267))).all()


def test_first_derivative(barrel):
    d1 = barrel.get_first_derivative()
    assert isinstance(d1, np.ndarray)


def test_second_derivative(barrel):
    d2 = barrel.get_second_derivative()
    assert isinstance(d2, np.ndarray)


def test_max(barrel):
    m = barrel.max()
    assert isinstance(m, np.ndarray)


def test_min(barrel):
    m = barrel.min()
    assert isinstance(m, np.ndarray)


def test_mean(barrel):
    m = barrel.mean()
    assert isinstance(m, np.ndarray)


def test_d1_mean(barrel):
    m = barrel.first_derivative_mean()
    assert isinstance(m, np.ndarray)


def test_d2_mean(barrel):
    m = barrel.second_derivative_mean()
    assert isinstance(m, np.ndarray)


def test_std(barrel):
    std = barrel.std()
    assert isinstance(std, np.ndarray)


def test_d1_std(barrel):
    std = barrel.first_derivative_std()
    assert isinstance(std, np.ndarray)


def test_d2_std(barrel):
    std = barrel.second_derivative_std()
    assert isinstance(std, np.ndarray)


def test_skewness(barrel):
    skew = barrel.skewness()
    assert isinstance(skew, np.ndarray)


def test_d1_skewness(barrel):
    skew = barrel.first_derivative_skewness()
    assert isinstance(skew, np.ndarray)


def test_d2_skewness(barrel):
    skew = barrel.second_derivative_skewness()
    assert isinstance(skew, np.ndarray)


def test_kurtosis(barrel):
    kurt = barrel.kurtosis()
    assert isinstance(kurt, np.ndarray)


def test_d1_kurt(barrel):
    kurt = barrel.first_derivative_kurtosis()
    assert isinstance(kurt, np.ndarray)


def test_d2_kurt(barrel):
    kurt = barrel.second_derivative_kurtosis()
    assert isinstance(kurt, np.ndarray)


def test_q1(barrel):
    q1 = barrel.first_quartile()
    assert isinstance(q1, np.ndarray)


def test_q2(barrel):
    q2 = barrel.second_quartile()
    assert isinstance(q2, np.ndarray)


def test_q3(barrel):
    q3 = barrel.third_quartile()
    assert isinstance(q3, np.ndarray)


def test_q2_q1_range(barrel):
    q2q1range = barrel.q2_q1_range()
    assert isinstance(q2q1range, np.ndarray)


def test_q3_q2_range(barrel):
    q3q2range = barrel.q3_q2_range()
    assert isinstance(q3q2range, np.ndarray)


def test_q3_q1_range(barrel):
    q3q1range = barrel.q3_q1_range()
    assert isinstance(q3q1range, np.ndarray)


def test_percentile_1(barrel):
    percentile_1 = barrel.percentile_1()
    assert isinstance(percentile_1, np.ndarray)


def test_percentile_99(barrel):
    percentile_99 = barrel.percentile_99()
    assert isinstance(percentile_99, np.ndarray)


def test_percentile_1_99_range(barrel):
    r = barrel.percentile_1_99_range()
    assert isinstance(r, np.ndarray)


def test_linear_regression_offset(barrel):
    x0 = barrel.linear_regression_offset()
    assert isinstance(x0, np.ndarray)


def test_linear_regression_slope(barrel):
    x1 = barrel.linear_regression_slope()
    assert isinstance(x1, np.ndarray)


def test_linear_regression_mse(barrel):
    mse = barrel.linear_regression_mse()
    assert isinstance(mse, np.ndarray)
