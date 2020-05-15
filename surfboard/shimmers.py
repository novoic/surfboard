#!/usr/bin/env python
"""This file contains all the functions needed to compute the shimmers of a waveform."""

import numpy as np

from .jitters import validate_frequencies

from .utils import (
    shifted_sequence,
    peak_amplitude_slidingwindow,
)


def validate_amplitudes(amplitudes, frequencies, max_a_factor, p_floor, p_ceil, max_p_factor):
    """First check that frequencies corresponding to this set of amplitudes are valid. Then
    Returns True if this set of amplitudes is validated as per the maximum
    amplitude factor principle, i.e. if amplitudes = [a1, a2, ... , an], this
    functions returns false if any two successive amplitudes alpha, beta satisfy
    alpha / beta > max_a_factor or beta / alpha > max_a_factor. False otherwise.

    Args:
        amplitudes (list): ordered list of amplitudes to run by this principle.
        frequencies (sequence, eg list, of floats): sequence of frequencies == 1 / period.
        max_a_factor (float): the threshold to run the principle.
        p_floor (float): minimum acceptable period.
        p_ceil (float): maximum acceptable period.
        max_p_factor (float): value to use for the period factor principle

    Returns:
        bool: True if this set of amplitudes satisifies the principle
            and this set of frequencies satisfies the period condition,
            False otherwise.
    """
    # First check that these frequencies satisfy the periods principle.
    if not validate_frequencies(frequencies, p_floor, p_ceil, max_p_factor):
        return False

    if max_a_factor is not None:
        for amp1, amp2 in zip(amplitudes[:-1], amplitudes[1:]):
            if amp1 / amp2 > max_a_factor or amp2 / amp1 > max_a_factor:
                return False
    return True


def get_local_shimmer(amplitudes, frequencies, max_a_factor, p_floor, p_ceil, max_p_factor):
    """Given a list of amplitudes, returns the localShimmer as per
    https://royalsocietypublishing.org/action/downloadSupplement?doi=10.1098%2Frsif.2010.0456&file=rsif20100456supp1.pdf

    Args:
        amplitudes (list of floats): The list of peak amplitudes in each frame.
        max_a_factor (float): The maximum A factor to validate amplitudes. See
            validate_amplitudes().

    Returns:
        float: The local shimmer computed over this sequence of amplitudes.
    """
    cumsum = 0
    counter = 0
    for (freq1, freq2), (amp1, amp2) in zip(shifted_sequence(frequencies, 2), shifted_sequence(amplitudes, 2)):
        if validate_amplitudes([amp1, amp2], [freq1, freq2], max_a_factor, p_floor, p_ceil, max_p_factor):
            cumsum += np.abs(amp1 - amp2)
            counter += 1

    mean_amplitude = np.mean(amplitudes)
    if counter != 0:
        local_shimmer = (cumsum / counter) / mean_amplitude if mean_amplitude != 0 else None
        return local_shimmer
    return None


def get_local_db_shimmer(amplitudes, frequencies, max_a_factor, p_floor, p_ceil, max_p_factor):
    """Given a list of amplitudes, returns the localdbShimmer as per
    https://royalsocietypublishing.org/action/downloadSupplement?doi=10.1098%2Frsif.2010.0456&file=rsif20100456supp1.pdf

    Args:
        amplitudes (list of floats): The list of peak amplitudes in each frame.
        max_a_factor (float): The maximum A factor to validate amplitudes. See
            validate_amplitudes().

    Returns:
        float: The local DB shimmer computed over this sequence of amplitudes.
    """
    cumsum = 0
    counter = 0
    for (freq1, freq2), (amp1, amp2) in zip(shifted_sequence(frequencies, 2), shifted_sequence(amplitudes, 2)):
        if validate_amplitudes([amp1, amp2], [freq1, freq2], max_a_factor, p_floor, p_ceil, max_p_factor):
            cumsum += np.abs(20 * np.log10(amp2 / amp1))
            counter += 1

    local_db_shimmer = cumsum / counter if counter != 0 else None
    return local_db_shimmer


def get_apq_shimmer(amplitudes, frequencies, max_a_factor, p_floor, p_ceil, max_p_factor, apq_no):
    """Given a list of amplitudes, returns the apq{apq_no}Shimmer as per
    https://royalsocietypublishing.org/action/downloadSupplement?doi=10.1098%2Frsif.2010.0456&file=rsif20100456supp1.pdf

    Args:
        amplitudes (list of floats): The list of peak amplitudes in each frame.
        max_a_factor (float): The maximum A factor to validate amplitudes. See
            validate_amplitudes().
        apq_no (int): an odd number which corresponds to the number of neighbors
            used to compute the shimmer.

    Returns:
        float: The apqShimmer computed over this sequence of amplitudes
            with this APQ number.
    """
    assert apq_no % 2 == 1, "To compute these, must have an odd APQ number."
    counter = 0
    cumsum = 0

    for freqs, amps in zip(shifted_sequence(frequencies, apq_no), shifted_sequence(amplitudes, apq_no)):
        if validate_amplitudes(amps, freqs, max_a_factor, p_floor, p_ceil, max_p_factor):
            counter += 1
            # apq_no is always odd, so int((apq_no - 1) / 2) doesn't assume which integer to return.
            cumsum += np.abs(amps[int((apq_no - 1) / 2)] - np.sum(amps) / apq_no)
    mean_amplitude = np.mean(amplitudes)
    if counter != 0:
        apq_shimmer = (cumsum / counter) / mean_amplitude if mean_amplitude != 0 else None
        return apq_shimmer
    return None


def get_shimmers(
    waveform, sample_rate, f0_contour, max_a_factor=1.6, p_floor=0.0001,
    p_ceil=0.02, max_p_factor=1.3
):
    """Compute five different types of shimmers using functions defined above.

    Args:
        waveform (np.array, [T, ]): waveform over which to compute shimmers
        sample_rate (int): sampling rate of waveform.
        f0_contour (np.array, [T / hop_length, ]): the fundamental frequency contour.
        max_a_factor (float): value to use for amplitude factor principle
        p_floor (float): minimum acceptable period.
        p_ceil (float): maximum acceptable period.
        max_p_factor (float): value to use for the period factor principle

    Returns:
        dict: Dictionary mapping strings to floats, with keys
            "localShimmer", "localdbShimmer", "apq3Shimmer", "apq5Shimmer",
            "apq11Shimmer"
    """
    amplitudes = peak_amplitude_slidingwindow(waveform, sample_rate)[0]

    local_shimmer = get_local_shimmer(
        amplitudes, f0_contour, max_a_factor, p_floor, p_ceil, max_p_factor,
    )
    local_db_shimmer = get_local_db_shimmer(
        amplitudes, f0_contour, max_a_factor, p_floor, p_ceil, max_p_factor,
    )
    apq3_shimmer = get_apq_shimmer(
        amplitudes, f0_contour, max_a_factor, p_floor, p_ceil, max_p_factor, apq_no=3
    )
    apq5_shimmer = get_apq_shimmer(
        amplitudes, f0_contour, max_a_factor, p_floor, p_ceil, max_p_factor, apq_no=5
    )
    apq11_shimmer = get_apq_shimmer(
        amplitudes, f0_contour, max_a_factor, p_floor, p_ceil, max_p_factor, apq_no=11
    )

    shimmers_dict = {
        "localShimmer": local_shimmer,
        "localdbShimmer": local_db_shimmer,
        "apq3Shimmer": apq3_shimmer,
        "apq5Shimmer": apq5_shimmer,
        "apq11Shimmer": apq11_shimmer,
    }

    return shimmers_dict
