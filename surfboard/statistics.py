"""This file contains the class which computes statistics from numpy arrays to turn components into features."""

from scipy.stats import (
    kurtosis,
    skew,
)

import numpy as np


class Barrel:
    """This class is used to instantiate components computed in the surfboard package.
    It helps us compute statistics on these components.
    """
    def __init__(self, component):
        """Instantiate a barrel with a component. Note that we require the component to be
        an np array. The first dimension represents the number of output features.
        and the second dimension represents time.
        
        Args:
            component (np.array, [n_feats, T]):
        """
        assert isinstance(component, np.ndarray), 'Barrels are instantiated with np arrays.'
        assert len(component.shape) == 2, f'Barrels must have shape [n_feats, T]. component is {component.shape}'

        self.type = type(component)
        self.component = component

    def __call__(self):
        return self.component

    def compute_statistics(self, statistic_list):
        """Compute statistics on self.component using a list of strings which identify
        which statistics to compute.

        Args:
            statistic_list (list of str): list of strings representing Barrel methods to
                be called.

        Returns:
            dict: Dictionary mapping str to float.
        """
        statistic_outputs = {}
        for statistic in statistic_list:
            method_to_call = getattr(self, statistic)
            result = method_to_call()
            # Case of only one component.
            if len(result) == 1:
                statistic_outputs[statistic] = float(result)
            else:
                for i, value in enumerate(result):
                    statistic_outputs[f"{statistic}_{i + 1}"] = value
        return statistic_outputs

    def get_first_derivative(self):
        """Compute the "first derivative" of self.component.
        Remember that self.component is of the shape [n_feats, T].

        Returns:
            np.array, [n_feats, T - 1]: First empirical derivative.
        """
        delta = self.component[:, 1:] - self.component[:, :-1]
        return delta

    def get_second_derivative(self):
        """Compute the "second derivative" of self.component.
        Remember that self.component is of the shape [n_feats, T].

        Returns:
            np.array, [n_feats, T - 2]: second empirical derivative.
        """
        delta = self.get_first_derivative()
        delta2 = delta[:, 1:] - delta[:, :-1]
        return delta2

    def max(self):
        """Compute the max of self.component on the last dimensions.

        Returns:
            np.array, [n_feats, ]: The maximum of each individual dimension
                in self.component
        """
        return np.max(self.component, -1)

    def min(self):
        """Compute the min of self.component on the last dimension.

        Returns:
            np.array, [n_feats, ]: The minimum of each individual dimension
                in self.component
        """
        return np.min(self.component, -1)

    def mean(self):
        """Compute the mean of self.component on the last dimension (time).

        Returns:
            np.array, [n_feats, ]: The mean of each individual dimension
                in self.component
        """
        return np.mean(self.component, -1)

    def first_derivative_mean(self):
        """Compute the mean of the first empirical derivative (delta coefficient)
        on the last dimension (time).

        Returns:
            np.array, [n_feats, ]: The mean of the first delta coefficient
             of each individual dimension in self.component
        """
        delta = self.get_first_derivative()
        return np.mean(delta, -1)

    def second_derivative_mean(self):
        """Compute the mean of the second empirical derivative (2nd delta coefficient)
        on the last dimension (time).

        Returns:
            np.array, [n_feats, ]: The mean of the second delta coefficient
             of each individual dimension in self.component
        """
        delta2 = self.get_second_derivative()
        return np.mean(delta2, -1)

    def std(self):
        """Compute the standard deviation of self.component on the last dimension
        (time).

        Returns:
            np.array, [n_feats, ]: The standard deviation of each individual
                dimension in self.component
        """
        return np.std(self.component, -1)

    def first_derivative_std(self):
        """Compute the std of the first empirical derivative (delta coefficient)
        on the last dimension (time).

        Returns:
            np.array, [n_feats, ]: The std of the first delta coefficient
             of each individual dimension in self.component
        """
        delta = self.get_first_derivative()
        return np.std(delta, -1)

    def second_derivative_std(self):
        """Compute the std of the second empirical derivative (2nd delta coefficient)
        on the last dimension (time).

        Returns:
            np.array, [n_feats, ]: The std of the second delta coefficient
             of each individual dimension in self.component
        """
        delta2 = self.get_second_derivative()
        return np.std(delta2, -1)

    def skewness(self):
        """Compute the skewness of self.component on the last dimension (time)

        Returns:
            np.array, [n_feats, ]: The skewness of each individual
                dimension in self.component
        """
        return skew(self.component, -1)

    def first_derivative_skewness(self):
        """Compute the skewness of the first empirical derivative (delta coefficient)
        on the last dimension (time).

        Returns:
            np.array, [n_feats, ]: The skewness of the first delta coefficient
             of each individual dimension in self.component
        """
        delta = self.get_first_derivative()
        return skew(delta, -1)

    def second_derivative_skewness(self):
        """Compute the skewness of the second empirical derivative (2nd delta coefficient)
        on the last dimension (time).

        Returns:
            np.array, [n_feats, ]: The skewness of the second delta coefficient
             of each individual dimension in self.component
        """
        delta2 = self.get_second_derivative()
        return skew(delta2, -1)

    def kurtosis(self):
        """Compute the kurtosis of self.component on the last dimension (time)

        Returns:
            np.array, [n_feats, ]: The kurtosis of each individual
                dimension in self.component
        """
        return kurtosis(self.component, -1)

    def first_derivative_kurtosis(self):
        """Compute the kurtosis of the first empirical derivative (delta coefficient)
        on the last dimension (time).

        Returns:
            np.array, [n_feats, ]: The kurtosis of the first delta coefficient
             of each individual dimension in self.component
        """
        delta = self.get_first_derivative()
        return kurtosis(delta, -1)

    def second_derivative_kurtosis(self):
        """Compute the kurtosis of the second empirical derivative (2nd delta coefficient)
        on the last dimension (time).

        Returns:
            np.array, [n_feats, ]: The kurtosis of the second delta coefficient
             of each individual dimension in self.component
        """
        delta2 = self.get_second_derivative()
        return kurtosis(delta2, -1)

    def first_quartile(self):
        """Compute the first quartile on the last dimension (time).

        Returns:
            np.array, [n_feats, ]: The first quartile of each individual dimension
                in self.component
        """
        return np.quantile(self.component, 0.25, axis=-1)

    def second_quartile(self):
        """Compute the second quartile on the last dimension (time). Same
        as the median.

        Returns:
            np.array, [n_feats, ]: The second quartile of each individual
                dimension in self.component (same as the median)
        """
        return np.quantile(self.component, 0.5, axis=-1)

    def third_quartile(self):
        """Compute the third quartile on the last dimension (time)

        Returns:
            np.array, [n_feats, ]: The third quartile of each individual
                dimension in self.component
        """
        return np.quantile(self.component, 0.75, axis=-1)

    def q2_q1_range(self):
        """Compute second and first quartiles. Return q2 - q1

        Returns:
            np.array, [n_feats, ]: The q2 - q1 range of each individual
                dimension in self.component
        """
        return self.second_quartile() - self.first_quartile()

    def q3_q2_range(self):
        """Compute third and second quartiles. Return q3 - q2

        Returns:
            np.array, [n_feats, ]: The q3 - q2 range of each individual
                dimension in self.component
        """
        return self.third_quartile() - self.second_quartile()

    def q3_q1_range(self):
        """Compute third and first quartiles. Return q3 - q1

        Returns:
            np.array, [n_feats, ]: The q3 - q1 range of each individual
                dimension in self.component
        """
        return self.third_quartile() - self.first_quartile()

    def percentile_1(self):
        """Compute the 1% percentile.

        Returns:
            np.array, [n_feats, ]: The 1st percentile of each individual
                dimension in self.component
        """
        return np.quantile(self.component, 0.01, axis=-1)

    def percentile_99(self):
        """Compute the 99% percentile.

        Returns:
            np.array, [n_feats, ]: The 99th percentile of each individual
                dimension in self.component
        """
        return np.quantile(self.component, 0.99, axis=-1)

    def percentile_1_99_range(self):
        """Compute 99% percentile and 1% percentile. Return the range.

        Returns:
            np.array, [n_feats, ]: The 99th - 1st percentile range of each
                individual dimension in self.component
        """
        return self.percentile_99() - self.percentile_1()

    def linear_regression_offset(self):
        """Consider each row of self.component as a time series over which we fit a line.
        Return the offset of that fitted line.

        Returns:
            np.array, [n_feats, ]: The linear regression offset of each
                individual dimension in self.component
        """
        _, offset = np.polyfit(
            np.arange(self.component.shape[-1]), self.component.T, deg=1
        )
        return offset

    def linear_regression_slope(self):
        """Consider each row of self.component as a time series over which we fit a line.
        Return the slope of that fitted line.

        Returns:
            np.array, [n_feats, ]: The linear regression slope of each
                individual dimension in self.component
        """
        slope, _ = np.polyfit(
            np.arange(self.component.shape[-1]), self.component.T, deg=1
        )
        return slope

    def linear_regression_mse(self):
        """Fit a line to the data. Compute the MSE.

        Returns:
            np.array, [n_feats, ]: The linear regression MSE of each
                individual dimension in self.component
        """
        slope, offset = np.polyfit(
            np.arange(self.component.shape[-1]), self.component.T, deg=1
        )
        linear_approximation = offset[:, np.newaxis] + slope[:, np.newaxis] \
            * np.array([np.arange(self.component.shape[-1]) for _ in range(self.component.shape[0])])

        mse = ((linear_approximation - self.component) ** 2).mean(-1)

        return mse
