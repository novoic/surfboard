#!/usr/bin/env python
"""This file contains components which do not fall under one category."""

import librosa
import pyloudnorm as pyln
import numpy as np

from scipy.stats import kurtosis
from scipy.signal import lfilter

from pysptk import swipe, rapt

from .utils import (
    metric_slidingwindow,
    numseconds_to_numsamples,
)


def get_crest_factor(waveform, sample_rate, rms, frame_length_seconds=0.04, hop_length_seconds=0.01):
    """Get the crest factor of this waveform, on sliding windows. This value measures the local intensity
    of peaks in a waveform. Implemented as per: https://en.wikipedia.org/wiki/Crest_factor

    Args:
        waveform (np.array, [T, ]): waveform over which to compute crest factor
        sample_rate (int > 0): number of samples per second in waveform
        rms (np.array, [1, T / hop_length]): energy values.
        frame_length_seconds (float): length of the sliding window, in seconds.
        hop_length_seconds (float): how much the window shifts for every timestep,
            in seconds.

    Returns:
        np.array, [1, T / hop_length]: Crest factor for each frame.
    """

    frame_length = numseconds_to_numsamples(frame_length_seconds, sample_rate)
    hop_length = numseconds_to_numsamples(hop_length_seconds, sample_rate)

    crest_factor_list = []
    # Iterate over values in rms. Each of these correspond to a window of the waveform.
    for i, rms_value in enumerate(rms[0]):
        waveform_window = waveform[i * hop_length: i * hop_length + frame_length]
        maxvalue = np.abs(waveform_window.max())
        crest_factor_list.append(maxvalue / rms_value)

    crest_factor = np.array(crest_factor_list)[np.newaxis, :]
    return crest_factor


def get_f0(
    waveform, sample_rate, hop_length_seconds=0.01, method='swipe', f0_min=60, f0_max=300
):
    """Compute the F0 contour using PYSPTK: https://github.com/r9y9/pysptk/.

    Args:
        waveform (np.array, [T, ]): waveform over which to compute f0
        sample_rate (int > 0): number of samples per second in waveform
        hop_length (int): hop size argument in pysptk.swipe. Corresponds to hopsize
            in the window sliding of the computation of f0.
        method (str): is one of 'swipe' or 'rapt'. Define which method to use for f0
            calculation. See https://github.com/r9y9/pysptk

    Returns:
        dict: Dictionary containing keys:
            "contour" (np.array, [1, t1]): f0 contour of waveform. Contains unvoiced
                frames.
            "values" (np.array, [1, t2]): nonzero f0 values waveform. Note that this
                discards all unvoiced frames. Use to compute mean, std, and other statistics.
            "mean" (float): mean of the f0 contour.
            "std" (float): standard deviation of the f0 contour.
    """
    assert method in ('swipe', 'rapt'), "The method argument should be one of 'swipe' or 'rapt'."

    hop_length = numseconds_to_numsamples(hop_length_seconds, sample_rate)
    if method == 'swipe':
        f0_contour = swipe(
            waveform.astype(np.float64),
            fs=sample_rate,
            hopsize=hop_length,
            min=f0_min,
            max=f0_max,
            otype="f0",
        )[np.newaxis, :]

    elif method == 'rapt':
        # For this estimation, waveform needs to be in the int PCM format.
        f0_contour = rapt(
            np.round(waveform * 32767).astype(np.float32),
            fs=sample_rate,
            hopsize=hop_length,
            min=f0_min,
            max=f0_max,
            otype="f0",
        )[np.newaxis, :]

    # Remove unvoiced frames.
    f0_values = f0_contour[:, np.where(f0_contour[0, :] != 0)][0]

    f0_mean = np.mean(f0_values[0])
    f0_std = np.std(f0_values[0])
    return {
        "contour": f0_contour,
        "values": f0_values,
        "mean": f0_mean,
        "std": f0_std,
    }


def get_ppe(rat_f0):
    """Compute pitch period entropy. Here is a reference MATLAB implementation:
    https://github.com/Mak-Sim/Troparion/blob/5126f434b96e0c1a4a41fa99dd9148f3c959cfac/Perturbation_analysis/pitch_period_entropy.m
    Note that computing the PPE relies on the existence of voiced portions in the F0 trajectory.

    Args:
        rat_f0 (np.array): f0 voiced frames divided by f_min

    Returns:
        float: The pitch period entropy, as per http://www.maxlittle.net/students/thesis_tsanas.pdf
    """
    semitone_f0 = np.log(rat_f0) / np.log(2 ** (1 / 12))

    # Whitening
    coefficients = librosa.core.lpc(semitone_f0[0], 2)
    semi_f0 = lfilter(coefficients, [1], semitone_f0)[0]
    # Filter to the [-1.5, 1.5] range.
    semi_f0 = semi_f0[np.where(semi_f0 > -1.5)]
    semi_f0 = semi_f0[np.where(semi_f0 < 1.5)]

    distrib = np.histogram(semi_f0, bins=30, density=True)[0]
    # Remove empty bins as these break the entropy calculation.
    distrib = distrib[distrib != 0]

    # Discrete probability distribution
    ppe = np.sum(-distrib * (np.log(distrib) / np.log(2)))
    return ppe


def get_shannon_entropy(sequence):
    """Given a sequence, compute the Shannon Entropy, defined in
    https://ijssst.info/Vol-16/No-4/data/8258a127.pdf

    Args:
        sequence (np.array, [t, ]): sequence over which to compute.

    Returns:
        float: shannon entropy.
    """
    # Remove zero entries in sequence as entropy is undefined.
    sequence = sequence[sequence != 0]
    return float((-(sequence ** 2) * np.log(sequence ** 2)).sum())


def get_shannon_entropy_slidingwindow(waveform, sample_rate, frame_length_seconds=0.04, hop_length_seconds=0.01):
    """Same function as above, but decorated by the metric_slidingwindow decorator.
    See above for documentation on this.

    Args:
        waveform (np.array, [T, ]): waveform over which to compute the shannon entropy array
        sample_rate (int > 0): number of samples per second in waveform
        frame_length_seconds (float): length of the sliding window, in seconds.
        hop_length_seconds (float): how much the window shifts for every timestep,
            in seconds.

    Returns:
        np.array, [1, T/hop_length]: Shannon entropy over windows.
    """
    frame_length = numseconds_to_numsamples(frame_length_seconds, sample_rate)
    hop_length = numseconds_to_numsamples(hop_length_seconds, sample_rate)

    @metric_slidingwindow(frame_length=frame_length, hop_length=hop_length)
    def new_shannon_entropy(waveform):
        return get_shannon_entropy(waveform)

    return new_shannon_entropy(waveform)


def get_loudness(waveform, sample_rate):
    """Compute the loudness of waveform using the pyloudnorm package.
    See https://github.com/csteinmetz1/pyloudnorm for more details on potential
    arguments to the functions below.

    Args:
        waveform (np.array, [T, ]): waveform to compute loudness on
        sample_rate (int > 0): sampling rate of waveform

    Returns:
        float: the loudness of self.waveform
    """
    meter = pyln.Meter(sample_rate)
    return meter.integrated_loudness(waveform)


def get_loudness_slidingwindow(waveform, sample_rate, frame_length_seconds=0.04, hop_length_seconds=0.01):
    """Same function as get_loudness, but decorated by the metric_slidingwindow decorator.
    See get_loudness documentation for this.

    Args:
        waveform (np.array, [T, ]): waveform over which to compute the kurtosis array
        sample_rate (int > 0): number of samples per second in waveform
        frame_length_seconds (float): length of the sliding window, in seconds.
        hop_length_seconds (float): how much the window shifts for every timestep,
            in seconds.

    Returns:
        np.array, [1, T / hop_length]: Frame level loudness
    """
    frame_length = numseconds_to_numsamples(frame_length_seconds, sample_rate)
    hop_length = numseconds_to_numsamples(hop_length_seconds, sample_rate)

    # We use truncate end = True here because the loudness calculation fails on short sequences.
    @metric_slidingwindow(frame_length=frame_length, hop_length=hop_length, truncate_end=True)
    def new_loudness(waveform):
        return get_loudness(waveform, sample_rate)

    return new_loudness(waveform)


def get_kurtosis_slidingwindow(waveform, sample_rate, frame_length_seconds=0.04, hop_length_seconds=0.01):
    """Same function as above, but decorated by the metric_slidingwindow decorator.
    See above documentation for this.

    Args:
        waveform (np.array, [T, ]): waveform over which to compute the kurtosis array
        sample_rate (int > 0): number of samples per second in waveform
        frame_length_seconds (float): length of the sliding window, in seconds.
        hop_length_seconds (float): how much the window shifts for every timestep,
            in seconds.

    Returns:
        np.array, [1, T/hop_length]: Kurtosis over windows
    """
    frame_length = numseconds_to_numsamples(frame_length_seconds, sample_rate)
    hop_length = numseconds_to_numsamples(hop_length_seconds, sample_rate)

    @metric_slidingwindow(frame_length=frame_length, hop_length=hop_length)
    def new_kurtosis(waveform):
        return kurtosis(waveform)

    return new_kurtosis(waveform)


def get_log_energy(matrix, time_axis=-1):
    """Compute the log energy of a matrix as per Abeyrante et al. 2013.

    Args:
        matrix (np.array): matrix over which to compute. This
            has to be a 1 or 2-dimensional np.array
        time_axis (int >= 0): the axis in matrix which corresponds
            to time.

    Returns:
        float: The log energy of matrix, computed as per
            the paper above.
    """
    assert len(matrix.shape) <= 2, "This function only works on 1d or 2d signals."
    return 10 * np.log10(1e-9 + np.sum(matrix ** 2, time_axis) / matrix.shape[time_axis])


def get_log_energy_slidingwindow(waveform, sample_rate, frame_length_seconds=0.04, hop_length_seconds=0.01):
    """Same function as above, but decorated by the metric_slidingwindow decorator.
    See above documentation for this.

    Args:
        waveform (np.array, [T, ]): waveform over which to compute the log energy array
        sample_rate (int > 0): number of samples per second in waveform
        frame_length_seconds (float): length of the sliding window, in seconds.
        hop_length_seconds (float): how much the window shifts for every timestep,
            in seconds.

    Returns:
        np.array, [1, T/hop_length]: log_energy over windows
    """
    frame_length = numseconds_to_numsamples(frame_length_seconds, sample_rate)
    hop_length = numseconds_to_numsamples(hop_length_seconds, sample_rate)

    @metric_slidingwindow(frame_length=frame_length, hop_length=hop_length)
    def new_log_energy(waveform):
        return get_log_energy(waveform)

    return new_log_energy(waveform)


def get_bark_spectrogram(waveform, sample_rate, n_fft_seconds, hop_length_seconds):
    """Convert a spectrogram to a bark-band spectrogram.
    
    Args:
        waveform (np.array, [T, ]): waveform over which to compute the bark
            spectrogram.
        sample_rate (int > 0): number of samples per second in waveform.
        n_fft_seconds (float > 0): length of the fft window, in seconds

    Returns:
        np.array, [n_bark_bands, t]: The original spectrogram
            with bins converted into the Bark scale.
    """
    bark_bands = [
        100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270,
        1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300,
        6400, 7700, 9500, 12000, 15500,
    ]

    n_fft = numseconds_to_numsamples(n_fft_seconds, sample_rate)
    hop_length = numseconds_to_numsamples(hop_length_seconds, sample_rate)

    # [n_frequency_bins, t]
    spectrogram, _ = librosa.core.spectrum._spectrogram(waveform, n_fft=n_fft, hop_length=hop_length)
    frequencies = librosa.core.fft_frequencies(sr=sample_rate, n_fft=n_fft)

    assert spectrogram.shape[0] == frequencies.shape[0], "Different number of frequencies..."

    # Initialise the output. It will be of shape [n_bark_bands, t]
    output = np.zeros((len(bark_bands), spectrogram.shape[1]), dtype=spectrogram.dtype)

    for band_idx in range(len(bark_bands) - 1):
        # Sum everything that falls in this bucket.
        output[band_idx] = np.sum(
            spectrogram[((frequencies >= bark_bands[band_idx]) & (frequencies < bark_bands[band_idx + 1]))], axis=0
        )

    return output
