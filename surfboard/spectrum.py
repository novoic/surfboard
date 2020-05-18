"""Spectrum features. The code in this file is inspired by audiocontentanalysis.org
For more details, visit the pyACA package: https://github.com/alexanderlerch/pyACA
"""

import numpy as np


def summed_magnitude_spectrum(magnitude_spectrum, keepdims=True):
    summed_magnitude_spectrum = magnitude_spectrum.sum(0, keepdims=keepdims)
    summed_magnitude_spectrum[summed_magnitude_spectrum == 0] = 1
    return summed_magnitude_spectrum


def get_spectral_centroid(magnitude_spectrum, sample_rate):
    """Given the magnitude spectrum and the sample rate of the
    waveform from which it came, compute the spectral centroid.

    Args:
        magnitude_spectrum (np.array, [n_frequencies, T / hop_length]): the spectrogram
        sample_rate (int): The sample rate of the waveform

    Returns:
        np.array [1, T / hop_length]: the spectral centroid sequence in Hz.
    """
    time_dimension = magnitude_spectrum.shape[0]
    summed = summed_magnitude_spectrum(magnitude_spectrum)
    return np.squeeze(
        np.dot(
            np.arange(0, time_dimension), magnitude_spectrum
        ) / summed / (time_dimension - 1) * sample_rate / 2
    )[np.newaxis, :]


def get_spectral_slope(magnitude_spectrum, sample_rate):
    """Given the magnitude spectrum and the sample rate of the
    waveform from which it came, compute the spectral slope.

    Args:
        magnitude_spectrum (np.array, [n_frequencies, T / hop_length]): the spectrogram
        sample_rate (int): The sample rate of the waveform

    Returns:
        np.array [1, T / hop_length]: the spectral slope sequence.
    """
    time_dimension = magnitude_spectrum.shape[0]
    spectrum_mean = magnitude_spectrum.mean(0, keepdims=True)

    centralized_spectrum = magnitude_spectrum - spectrum_mean
    index = np.arange(0, time_dimension) - time_dimension / 2
    slope = np.dot(index, centralized_spectrum) / np.dot(index, index)
    return slope[np.newaxis, :]


def get_spectral_flux(magnitude_spectrum, sample_rate):
    """Given the magnitude spectrum and the sample rate of the
    waveform from which it came, compute the spectral flux.

    Args:
        magnitude_spectrum (np.array, [n_frequencies, T / hop_length]): the spectrogram
        sample_rate (int): The sample rate of the waveform

    Returns:
        np.array [1, T / hop_length]: the spectral flux sequence.
    """
    first_column = magnitude_spectrum[:, 0][:, np.newaxis]

    # Replicate first column to set first delta coeff to 0.
    new_magnitude_spectrum = np.concatenate(
        (first_column, magnitude_spectrum), axis=-1
    )
    delta_coefficient = np.diff(new_magnitude_spectrum, 1, axis=1)

    flux = np.sqrt(
        (delta_coefficient ** 2).sum(0)
    ) / new_magnitude_spectrum.shape[0]
    return flux[np.newaxis, :]


def get_spectral_spread(magnitude_spectrum, sample_rate):
    """Given the magnitude spectrum and the sample rate of the
    waveform from which it came, compute the spectral spread.

    Args:
        magnitude_spectrum (np.array, [n_frequencies, T / hop_length]): the spectrogram
        sample_rate (int): The sample rate of the waveform

    Returns:
        np.array [1, T / hop_length]: the spectral spread (Hz).
    """
    summed = summed_magnitude_spectrum(magnitude_spectrum, keepdims=False)
    n_frequencies, n_timesteps = magnitude_spectrum.shape

    Hz_scaling_factor = 2 * (n_frequencies - 1) / sample_rate

    spectral_centroids = get_spectral_centroid(magnitude_spectrum, sample_rate)[0]
    scaled_centroids = spectral_centroids * Hz_scaling_factor

    index_iter = np.arange(0, n_frequencies)
    spectral_spread = np.array([
        np.sqrt(
            np.dot((index_iter - centroid) ** 2, magnitude_spectrum[:, time_index]) / summed[time_index]
        ) for time_index, centroid in enumerate(scaled_centroids)
    ])
    # Conversion back to Hz.
    return (spectral_spread / Hz_scaling_factor)[np.newaxis, :]


def get_spectral_skewness(magnitude_spectrum, sample_rate):
    """Given the magnitude spectrum and the sample rate of the
    waveform from which it came, compute the spectral skewness.

    Args:
        magnitude_spectrum (np.array, [n_frequencies, T / hop_length]): the spectrogram
        sample_rate (int): The sample rate of the waveform

    Returns:
        np.array [1, T / hop_length]: the spectral skewness.
    """
    n_frequencies, n_timesteps = magnitude_spectrum.shape
    Hz_scaling_factor = 2 * (n_frequencies - 1) / sample_rate
    summed = summed_magnitude_spectrum(magnitude_spectrum, keepdims=False)

    spectral_centroids = get_spectral_centroid(magnitude_spectrum, sample_rate)[0]
    spectral_spreads = get_spectral_spread(magnitude_spectrum, sample_rate)[0]
    # Replace zero spreads by 1.
    spectral_spreads[spectral_spreads == 0] = 1
    index_iter = np.arange(0, n_frequencies) / Hz_scaling_factor

    spectral_skewness = [
        np.dot((index_iter - centroid) ** 3, magnitude_spectrum[:, time_index]) / (
            n_frequencies * spread ** 3 * summed[time_index]
        ) for time_index, (centroid, spread) in enumerate(zip(spectral_centroids, spectral_spreads))
    ]
    return np.array(spectral_skewness)[np.newaxis, :]


def get_spectral_kurtosis(magnitude_spectrum, sample_rate):
    """Given the magnitude spectrum and the sample rate of the
    waveform from which it came, compute the spectral skewness.

    Args:
        magnitude_spectrum (np.array, [n_frequencies, T / hop_length]): the spectrogram
        sample_rate (int): The sample rate of the waveform

    Returns:
        np.array [1, T / hop_length]: the spectral kurtosis.
    """
    n_frequencies, n_timesteps = magnitude_spectrum.shape
    Hz_scaling_factor = 2 * (n_frequencies - 1) / sample_rate
    summed = summed_magnitude_spectrum(magnitude_spectrum, keepdims=False)

    spectral_centroids = get_spectral_centroid(magnitude_spectrum, sample_rate)[0]
    spectral_spreads = get_spectral_spread(magnitude_spectrum, sample_rate)[0]
    # Replace zero spreads by 1.
    spectral_spreads[spectral_spreads == 0] = 1
    index_iter = np.arange(0, n_frequencies) / Hz_scaling_factor

    spectral_skewness = [
        np.dot((index_iter - centroid) ** 4, magnitude_spectrum[:, time_index]) / (
            n_frequencies * spread ** 4 * summed[time_index]
        ) for time_index, (centroid, spread) in enumerate(zip(spectral_centroids, spectral_spreads))
    ]
    return (np.array(spectral_skewness) - 3)[np.newaxis, :]
