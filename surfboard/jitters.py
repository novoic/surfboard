#!/usr/bin/env python
"""This file contains all the functions needed to compute the jitters of a waveform."""

import numpy as np


from .utils import (
    shifted_sequence,
)


def validate_frequencies(frequencies, p_floor, p_ceil, max_p_factor):
    """Given a sequence of frequencies, [f1, f2, ..., fn], a minimum period,
    maximum period, and maximum period factor, first remove all frequencies computed as 0.
    Then, if periods are the inverse frequencies, this function returns
    True if the sequence of periods satisfies the conditions, otherwise
    returns False. In order to satisfy the maximum period factor, the periods
    have to satisfy pi / pi+1 < max_p_factor and pi+1 / pi < max_p_factor.

    Args:
        frequencies (sequence, eg list, of floats): sequence of frequencies == 1 / period.
        p_floor (float): minimum acceptable period.
        p_ceil (float): maximum acceptable period.
        max_p_factor (float): value to use for the period factor principle

    Returns:
        bool: True if the conditions are met, False otherwise.
    """
    for freq in frequencies:
        if freq == 0:
            return False

    periods = [1 / f for f in frequencies]
    for period in periods:
        if period < p_floor or period > p_ceil:
            return False

    if len(periods) > 1 and max_p_factor is not None:
        for period1, period2 in zip(periods[:-1], periods[1:]):
            if period1 / period2 > max_p_factor or period2 / period1 > max_p_factor:
                return False
    return True


def get_mean_period(frequencies, p_floor, p_ceil, max_p_factor):
    """Given a sequence of frequencies, passes these through the validation phase,
    then computes the mean of the remaining periods. Note period = 1/f.

    Args:
        frequencies (sequence, eg list, of floats):  sequence of frequencies
        p_floor (float): minimum acceptable period.
        p_ceil (float): maximum acceptable period.
        max_p_factor (float): value to use for the period factor principle

    Returns:
        float: The mean of the acceptable periods.
    """
    cumsum = 0
    counter = 0
    for freq in frequencies:
        if validate_frequencies([freq], p_floor, p_ceil, max_p_factor):
            cumsum += 1 / freq
            counter += 1
    mean_period = cumsum / counter if counter != 0 else None
    return mean_period


def get_local_absolute_jitter(frequencies, p_floor, p_ceil, max_p_factor):
    """Given a sequence of frequencies, and some period conditions,
    compute the local absolute jitter, as per 
    https://royalsocietypublishing.org/action/downloadSupplement?doi=10.1098%2Frsif.2010.0456&file=rsif20100456supp1.pdf

    Args:
        frequencies (sequence, eg list, of floats): sequence of estimated frequencies
        p_floor (float): minimum acceptable period.
        p_ceil (float): maximum acceptable period.
        max_p_factor (float): value to use for the period factor principle

    Returns:
        float: the local absolute jitter.
    """
    cumsum = 0
    counter = 0
    for pair in shifted_sequence(frequencies, 2):
        freq1, freq2 = pair
        if validate_frequencies([freq1, freq2], p_floor, p_ceil, max_p_factor):
            counter += 1
            cumsum += np.abs((1 / freq1) - (1 / freq2))

    return cumsum / counter if counter != 0 else None


def get_local_jitter(frequencies, p_floor, p_ceil, max_p_factor):
    """Given a sequence of frequencies, and some period conditions, compute the local
    jitter, as per https://royalsocietypublishing.org/action/downloadSupplement?doi=10.1098%2Frsif.2010.0456&file=rsif20100456supp1.pdf

    Args:
        frequencies (sequence, eg list, of floats): sequence of estimated frequencies
        p_floor (float): minimum acceptable period.
        p_ceil (float): maximum acceptable period.
        max_p_factor (float): value to use for the period factor principle

    Returns:
        float: the local jitter.
    """
    mean_period = get_mean_period(frequencies, p_floor, p_ceil, max_p_factor)
    local_absolute_jitter = get_local_absolute_jitter(frequencies, p_floor, p_ceil, max_p_factor)
    if mean_period is not None and local_absolute_jitter is not None:
        return local_absolute_jitter / mean_period if mean_period != 0 else None
    return None


def get_rap_jitter(frequencies, p_floor, p_ceil, max_p_factor):
    """Given a sequence of frequencies, and some period conditions,
    compute the rap jitter, as per 
    https://royalsocietypublishing.org/action/downloadSupplement?doi=10.1098%2Frsif.2010.0456&file=rsif20100456supp1.pdf

    Args:
        frequencies (sequence, eg list, of floats): sequence of estimated frequencies
        p_floor (float): minimum acceptable period.
        p_ceil (float): maximum acceptable period.
        max_p_factor (float): value to use for the period factor principle

    Returns:
        float: the rap jitter.
    """
    counter = 0
    cumsum = 0
    mean_period = get_mean_period(frequencies, p_floor, p_ceil, max_p_factor)

    for freq1, freq2, freq3 in shifted_sequence(frequencies, 3):
        if validate_frequencies([freq1, freq2, freq3], p_floor, p_ceil, max_p_factor):
            cumsum += np.abs(1 / freq2 - (1 / freq1 + 1 / freq2 + 1 / freq3) / 3)
            counter += 1

    if counter != 0:
        rap_jitter = (cumsum / counter) / mean_period if mean_period != 0 else None
        return rap_jitter
    return None


def get_ppq5_jitter(frequencies, p_floor, p_ceil, max_p_factor):
    """Given a sequence of frequencies, and some period conditions,
    compute the ppq5 jitter, as per 
    https://royalsocietypublishing.org/action/downloadSupplement?doi=10.1098%2Frsif.2010.0456&file=rsif20100456supp1.pdf

    Args:
        frequencies (sequence, eg list, of floats): sequence of estimated frequencies
        p_floor (float): minimum acceptable period.
        p_ceil (float): maximum acceptable period.
        max_p_factor (float): value to use for the period factor principle

    Returns:
        float: the ppq5 jitter.
    """
    counter = 0
    cumsum = 0
    mean_period = get_mean_period(frequencies, p_floor, p_ceil, max_p_factor)

    for freq1, freq2, freq3, freq4, freq5 in shifted_sequence(frequencies, 5):
        if validate_frequencies([freq1, freq2, freq3, freq4, freq5], p_floor, p_ceil, max_p_factor):
            counter += 1
            cumsum += np.abs(1 / freq3 - (1 / freq1 + 1 / freq2 + 1 / freq3 + 1 / freq4 + 1 / freq5) / 5)

    if counter != 0:
        ppq5_jitter = (cumsum / counter) / mean_period if mean_period != 0 else None
        return ppq5_jitter
    return None


def get_ddp_jitter(frequencies, p_floor, p_ceil, max_p_factor):
    """Given a sequence of frequencies, and some period conditions,
    compute the ddp jitter, as per 
    http://www.fon.hum.uva.nl/praat/manual/PointProcess__Get_jitter__ddp____.html

    Args:
        frequencies (sequence, eg list, of floats): sequence of estimated frequencies
        p_floor (float): minimum acceptable period.
        p_ceil (float): maximum acceptable period.
        max_p_factor (float): value to use for the period factor principle

    Returns:
        float: the ddp jitter.
    """
    counter = 0
    cumsum = 0
    mean_period = get_mean_period(frequencies, p_floor, p_ceil, max_p_factor)

    for freq1, freq2, freq3 in shifted_sequence(frequencies, 3):
        if validate_frequencies([freq1, freq2, freq3], p_floor, p_ceil, max_p_factor):
            counter += 1
            cumsum += np.abs((1 / freq3 - 1 / freq2) - (1 / freq2 - 1 / freq1))

    if counter != 0:
        ddp_jitter = (cumsum / counter) / mean_period if mean_period != 0 else None
        return ddp_jitter
    return None


def get_jitters(f0_contour, p_floor=0.0001, p_ceil=0.02, max_p_factor=1.3):
    """Compute the jitters mathematically, according to certain conditions
    given by p_floor, p_ceil and max_p_factor.

    Args:
        f0_contour (np.array [T / hop_length, ]): the fundamental frequency contour.
        p_floor (float): minimum acceptable period.
        p_ceil (float): maximum acceptable period.
        max_p_factor (float): value to use for the period factor principle

    Returns:
        dict: Dictionary mapping strings to floats, with keys
            "localJitter", "localabsoluteJitter", "rapJitter", "ppq5Jitter",
            "ddpJitter"
    """
    local_absolute_jitter = get_local_absolute_jitter(f0_contour, p_floor, p_ceil, max_p_factor)
    local_jitter = get_local_jitter(f0_contour, p_floor, p_ceil, max_p_factor)
    rap_jitter = get_rap_jitter(f0_contour, p_floor, p_ceil, max_p_factor)
    ppq5_jitter = get_ppq5_jitter(f0_contour, p_floor, p_ceil, max_p_factor)
    ddp_jitter = get_ddp_jitter(f0_contour, p_floor, p_ceil, max_p_factor)

    jitters_dict = {
        "localJitter": local_jitter,
        "localabsoluteJitter": local_absolute_jitter,
        "rapJitter": rap_jitter,
        "ppq5Jitter": ppq5_jitter,
        "ddpJitter": ddp_jitter,
    }

    return jitters_dict
