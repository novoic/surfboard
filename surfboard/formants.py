"""This code is inspired by the following repository:
https://github.com/manishmalik/Voice-Classification/blob/master/rootavish/formant.py
More importantly, by the following Matlab code:
https://uk.mathworks.com/help/signal/ug/formant-estimation-with-lpc-coefficients.html
The implementation is ours. These values have been validated against the corresponding methods in Praat and agree to a small error margin.
"""

import numpy as np
import math
from scipy.signal import lfilter
from librosa.core import lpc

from .utils import (
    metric_slidingwindow,
    numseconds_to_numsamples,
)

"""
Estimate formants using LPC.
"""


def estimate_formants_lpc(waveform, sample_rate, num_formants=5):
    hamming_win = np.hamming(len(waveform))
    # Apply window and high pass filter.
    x_win = waveform * hamming_win
    x_filt = lfilter([1], [1.0, 0.63], x_win)

    # Get LPC. From mathworks link above, the general rule is that the
    # order is two times the expected number of formants plus 2. We use
    # 5 as a base because we discard the formant f0 and want f1...f4.
    lpc_rep = lpc(x_filt, 2 + int(sample_rate / 1000))
    # Calculate the frequencies.
    roots = [r for r in np.roots(lpc_rep) if np.imag(r) >= 0]
    angles = np.arctan2(np.imag(roots), np.real(roots))

    return sorted(angles * (sample_rate / (2 * math.pi)))


def get_formants(waveform, sample_rate):
    """Estimate the first four formant frequencies using LPC

    Args:
        waveform (np.array, [T, ]): waveform over which to compute formants
        sample_rate (int): sampling rate of waveform

    Returns:
        dict: Dictionary mapping {'f1', 'f2', 'f3', 'f4'} to
            corresponding {first, second, third, fourth} formant frequency.
    """
    myformants = estimate_formants_lpc(waveform, sample_rate)
    formants_dict = {
        "f1": myformants[2],
        "f2": myformants[3],
        "f3": myformants[4],
        "f4": myformants[5],
    }
    return formants_dict


def get_unique_formant(waveform, sample_rate, formant):
    """Same as get_formants, but return in a np array.
    """
    dict_out = get_formants(waveform, sample_rate)
    return dict_out[formant]


def get_formants_slidingwindow(waveform, sample_rate, formant, frame_length_seconds=0.04, hop_length_seconds=0.01):
    """Apply the metric_slidingwindow decorator to the get_formants function above.
    We slightly change the get_formants in order to return a [4, T / hop_length] array instead
    of dictionaries.

    Args:
        waveform (np.array [T,]): waveform over which to compute.
        sample_rate (int): number of samples per second in the waveform
        frame_length_seconds (float): how many seconds in one frame. This
            value is defined in seconds instead of number of samples.
        hop_length_seconds (float): how many seconds frames shift each step.
            This value is defined in seconds instead of number of samples.

    Returns:
        np.array, [4, T / hop_length]: f1, f2, f3, f4 formants on each window.
    """
    frame_length = numseconds_to_numsamples(frame_length_seconds, sample_rate)
    hop_length = numseconds_to_numsamples(hop_length_seconds, sample_rate)

    @metric_slidingwindow(frame_length=frame_length, hop_length=hop_length)
    def new_get_unique_formant(waveform, sample_rate, formant):
        return get_unique_formant(waveform, sample_rate, formant)

    return new_get_unique_formant(waveform, sample_rate, formant)
