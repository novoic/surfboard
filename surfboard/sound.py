#!/usr/bin/env python
"""This file contains the central Waveform class of the surfboard package, and all the corresponding methods"""

import librosa
from scipy.signal import cwt, morlet
import numpy as np

from . import (
    jitters,
    shimmers,
    formants,
    spectrum,
    hnr,
    dfa,
)

from .misc_components import (
    get_shannon_entropy,
    get_shannon_entropy_slidingwindow,
    get_loudness,
    get_loudness_slidingwindow,
    get_kurtosis_slidingwindow,
    get_ppe,
    get_f0,
    get_crest_factor,
    get_log_energy,
    get_log_energy_slidingwindow,
    get_bark_spectrogram,
)

from .utils import (
    numseconds_to_numsamples,
    lpc_to_lsf,
    parse_component,
)


class Waveform:
    """
    The central class of the package. This class instantiates with a path to a sound file and
    a sample rate to load it or a signal and a sample rate. We can then use methods of this class to
    compute various components. 
    """

    def __init__(self, path=None, signal=None, sample_rate=44100):
        """
        Instantiate an object of this class. This loads the audio into a (T,) np.array: self.waveform.

            Args:
                path (str): Path to a sound file (eg .wav or .mp3).
                sample_rate (int): Sample rate used to load the sound file.
                OR:
                signal (np.array, [T, ]): Waveform signal.
                sample_rate (int): Sample rate of the waveform.
        """
        if signal is None:
            assert isinstance(path, str), "The path argument to the constructor must be a string."

        if path is None:
            assert isinstance(signal, np.ndarray), "The signal argument to the constructor must be a np.array."
            assert len(signal.shape) == 1, "The signal argument to the constructor must be a 1D [T, ] array."

        if (signal is not None) and (path is not None):
            raise ValueError("Cannot give both a path to a sound file and a signal. Take your pick!")

        assert isinstance(sample_rate, int), "The sample_rate argument to the constructor must be an integer."

        if path is not None:
            self._waveform = librosa.core.load(path, sr=sample_rate)[0]
        else:
            self._waveform = signal

        assert self.waveform.shape[0] > 1, "Your waveform must have more than one element."

        self._sample_rate = sample_rate

    # Make the variables instantiated in __init__ as properties: this hides them behind a "shadow attribute".
    @property
    def waveform(self):
        """Properties written in this way prevent users to assign to self.waveform"""
        return self._waveform

    @property
    def sample_rate(self):
        """Properties written in this way prevent users to assign to self.sample_rate"""
        return self._sample_rate

    def compute_components(self, component_list):
        """
        Compute components from self.waveform and self.sample_rate using a list of strings
        which identify which components to compute. You can pass in arguments to the 
        components (e.g. frame_length_seconds) by passing in the components as dictionaries.
        For example: {'mfcc': {'n_mfcc': 26}}. See README.md for more details.

        Args:
            component_list (list of str or dict): The methods to be computed.
                If elements are str, then the method uses default arguments.
                If dict, the arguments are passed to the methods.

        Returns:
            dict: Dictionary mapping component names to computed components.
        """
        components = {}
        for component in component_list:
            component_name, arguments = parse_component(component)
            try:
                method_to_call = getattr(self, component_name)
                if arguments is not None:
                    result = method_to_call(**arguments)
                else:
                    result = method_to_call()
            # Set result as None so as not to skip an entire Waveform object.
            except AttributeError:
                raise NotImplementedError(f'The component {component_name} does not exist.')
            except:
                result = float("nan")
            components[component_name] = result
        return components

    def mfcc(self, n_mfcc=13, n_fft_seconds=0.04, hop_length_seconds=0.01):
        """
        Given a number of MFCCs, use the librosa.feature.mfcc method to compute the correct
        number of MFCCs on self.waveform and returns the array.

        Args:
            n_mfcc (int): number of MFCCs to compute
            n_fft_seconds (float): length of the FFT window in seconds.
            hop_length_seconds (float): how much the window shifts for every timestep,
                in seconds.

        Returns:
            np.array, [n_mfcc, T / hop_length]: MFCCs.
        """
        n_fft = numseconds_to_numsamples(n_fft_seconds, self.sample_rate)
        hop_length = numseconds_to_numsamples(hop_length_seconds, self.sample_rate)

        return librosa.feature.mfcc(
            self.waveform, sr=self.sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length,
        )

    def log_melspec(self, n_mels=128, n_fft_seconds=0.04, hop_length_seconds=0.01):
        """Given a number of filter banks, this uses the librosa.feature.melspectrogram method to
        compute the log melspectrogram of self.waveform.

        Args:
            n_mels (int): Number of filter banks per time step in the log melspectrogram.
            n_fft_seconds (float): Length of the FFT window in seconds.
            hop_length_seconds (float): How much the window shifts for every timestep, in seconds.

        Returns:
            np.array, [n_mels, T_mels]: Log mel spectrogram.
        """
        n_fft = numseconds_to_numsamples(n_fft_seconds, self.sample_rate)
        hop_length = numseconds_to_numsamples(hop_length_seconds, self.sample_rate)

        melspec = librosa.feature.melspectrogram(
            self.waveform, sr=self.sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
        )
        return librosa.power_to_db(melspec, ref=np.max)

    def magnitude_spectrum(self, n_fft_seconds=0.04, hop_length_seconds=0.01):
        """Compute the STFT of self.waveform. This is used for further spectral analysis.

        Args:
            n_fft_seconds (float): Length of the FFT window in seconds.
            hop_length_seconds (float): How much the window shifts for every timestep, in seconds.

        Returns:
            np.array, [n_fft / 2 + 1, T / hop_length]: The magnitude spectrogram
        """
        n_fft = numseconds_to_numsamples(n_fft_seconds, self.sample_rate)
        hop_length = numseconds_to_numsamples(hop_length_seconds, self.sample_rate)

        mag_spectrum, _ = librosa.core.spectrum._spectrogram(self.waveform, n_fft=n_fft, hop_length=hop_length)
        return mag_spectrum

    def bark_spectrogram(self, n_fft_seconds=0.04, hop_length_seconds=0.01):
        """Compute the magnitude spectrum of self.waveform and arrange the frequency bins
        in the Bark scale. See https://en.wikipedia.org/wiki/Bark_scale

        Args:
            n_fft_seconds (float): Length of the FFT window in seconds.
            hop_length_seconds (float): How much the window shifts for every timestep, in seconds.

        Returns:
            np.array, [n_bark_bands, T / hop_length]: The Bark spectrogram
        """
        return get_bark_spectrogram(
            self.waveform, self.sample_rate, n_fft_seconds, hop_length_seconds,
        )

    def morlet_cwt(self, widths=None):
        """Compute the Morlet Continuous Wavelet Transform of self.waveform. Note that this
        method returns a large matrix. Shown relevant in Vasquez-Correa et Al, 2016.

        Args:
            wavelet (str): Wavelet to use. Currently only support "morlet".
            widhts (None or list): If None, uses default of 32 evenly spaced widths
                as [i * sample_rate / 500 for i in range(1, 33)]

        Returns:
            np.array, [len(widths), T]: The continuous wavelet transform
        """

        # This comes from having [32, 64, ..., 1024] at 16 kHz.
        if widths is None:
            widths = [int(i * self.sample_rate / 500) for i in range(1, 33)]

        # Take absolute value because the output of cwt is complex for morlet wavelet.
        return np.abs(cwt(self.waveform, morlet, widths))

    def chroma_stft(self, n_fft_seconds=0.04, hop_length_seconds=0.01, n_chroma=12):
        """See librosa.feature documentation for more details on this component. This computes
        a chromagram from a waveform.

        Args:
            n_fft_seconds (float): Length of the FFT window in seconds.
            hop_length_seconds (float): How much the window shifts for every timestep,
                in seconds.
            n_chroma (int): Number of chroma bins to compute.

        Returns:
            np.array, [n_chroma, T / hop_length]: The chromagram
        """
        n_fft = numseconds_to_numsamples(n_fft_seconds, self.sample_rate)
        hop_length = numseconds_to_numsamples(hop_length_seconds, self.sample_rate)

        assert isinstance(n_chroma, int) and n_chroma > 0, "n_chroma must be a >0 integer."

        return librosa.feature.chroma_stft(
            y=self.waveform, sr=self.sample_rate, n_fft=n_fft, hop_length=hop_length, n_chroma=n_chroma,
        )

    def chroma_cqt(self, hop_length_seconds=0.01, n_chroma=12):
        """See librosa.feature documentation for more details on this component. This computes
        a constant-Q chromagram from a waveform.

        Args:
            hop_length_seconds (float): How much the window shifts for every timestep,
                in seconds.
            n_chroma (int): Number of chroma bins to compute.

        Returns:
            np.array, [n_chroma, T / hop_length]: Constant-Q transform mode
        """
        hop_length = numseconds_to_numsamples(hop_length_seconds, self.sample_rate)

        assert isinstance(n_chroma, int) and n_chroma > 0, "n_chroma must be a >0 integer."

        return librosa.feature.chroma_cqt(
            y=self.waveform, sr=self.sample_rate, hop_length=hop_length, n_chroma=n_chroma,
        )

    def chroma_cens(self, hop_length_seconds=0.01, n_chroma=12):
        """See librosa.feature documentation for more details on this component. This computes
        the CENS chroma variant from a waveform.

        Args:
            hop_length_seconds (float): How much the window shifts for every timestep,
                in seconds.
            n_chroma (int): Number of chroma bins to compute.

        Returns:
            np.array, [n_chroma, T / hop_length]: CENS-chromagram
        """
        hop_length = numseconds_to_numsamples(hop_length_seconds, self.sample_rate)

        assert isinstance(n_chroma, int) and n_chroma > 0, "n_chroma must be a >0 integer."

        return librosa.feature.chroma_cens(
            y=self.waveform, sr=self.sample_rate, hop_length=hop_length, n_chroma=n_chroma,
        )

    def spectral_slope(self, n_fft_seconds=0.04, hop_length_seconds=0.01):
        """Compute the magnitude spectrum, and compute the spectral slope from that. This is a
        basic approximation of the spectrum by a linear regression line. There is one coefficient
        per timestep.

        Args:
            n_fft_seconds (float): Length of the FFT window in seconds.
            hop_length_seconds (float): How much the window shifts for every timestep,
                in seconds.

        Returns:
            np.array, [1, T / hop_length]: Linear regression slope, for every timestep.
        """
        magnitude_spectrum = self.magnitude_spectrum(n_fft_seconds, hop_length_seconds)

        return spectrum.get_spectral_slope(magnitude_spectrum, self.sample_rate)

    def spectral_flux(self, n_fft_seconds=0.04, hop_length_seconds=0.01):
        """Compute the magnitude spectrum, and compute the spectral flux from that. This is a
        basic metric, measuring the rate of change of the spectrum.

        Args:
            n_fft_seconds (float): Length of the FFT window in seconds.
            hop_length_seconds (float): How much the window shifts for every timestep,
                in seconds.

        Returns:
            np.array, [1, T / hop_length]: The spectral flux array.
        """
        magnitude_spectrum = self.magnitude_spectrum(n_fft_seconds, hop_length_seconds)

        return spectrum.get_spectral_flux(magnitude_spectrum, self.sample_rate)

    def spectral_entropy(self, n_fft_seconds=0.04, hop_length_seconds=0.01):
        """Compute the magnitude spectrum, and compute the spectral entropy from that. To compute
        that, simply normalize each frame of the spectrum, so that they are a probability
        distribution, then compute the entropy from that.

        Args:
            n_fft_seconds (float): Length of the FFT window in seconds.
            hop_length_seconds (float): How much the window shifts for every timestep,
                in seconds.

        Returns:
            np.array, [1, T / hop_length]: The entropy of each normalized frame.
        """
        # [n_frequency_bins, T / hop_length]
        magnitude_spectrum = self.magnitude_spectrum(n_fft_seconds, hop_length_seconds)

        # Normalize at each time frame. Compute the mean over the first dimension (i.e. sum of
        # each column).
        col_sums = magnitude_spectrum.sum(axis=0)
        normalized_spectrum = magnitude_spectrum / col_sums[np.newaxis, :]
        return (- normalized_spectrum * np.log(normalized_spectrum + 1e-9)).sum(0)[np.newaxis, :]

    def spectral_centroid(self, n_fft_seconds=0.04, hop_length_seconds=0.01):
        """Compute spectral centroid from magnitude spectrum. "First moment".

        Args:
            n_fft_seconds (float): Length of the FFT window in seconds.
            hop_length_seconds (float): How much the window shifts for every timestep,
                in seconds.

        Returns:
            np.array, [1, T / hop_length]: Spectral centroid of the magnitude spectrum (first moment).
        """
        magnitude_spectrum = self.magnitude_spectrum(n_fft_seconds, hop_length_seconds)

        return spectrum.get_spectral_centroid(magnitude_spectrum, self.sample_rate)

    def spectral_spread(self, n_fft_seconds=0.04, hop_length_seconds=0.01):
        """Compute spectral spread (also spectral variance) from magnitude spectrum.

        Args:
            n_fft_seconds (float): Length of the FFT window in seconds.
            hop_length_seconds (float): How much the window shifts for every timestep,
                in seconds.

        Returns:
            np.array, [1, T / hop_length: Spectral skewness of the magnitude spectrum (second moment).
        """
        magnitude_spectrum = self.magnitude_spectrum(n_fft_seconds, hop_length_seconds)

        return spectrum.get_spectral_spread(magnitude_spectrum, self.sample_rate)

    def spectral_skewness(self, n_fft_seconds=0.04, hop_length_seconds=0.01):
        """Compute spectral skewness from magnitude spectrum.

        Args:
            n_fft_seconds (float): Length of the FFT window in seconds.
            hop_length_seconds (float): How much the window shifts for every timestep,
                in seconds.

        Returns:
            np.array, [1, T / hop_length: Spectral skewness of the magnitude spectrum (third moment).
        """
        magnitude_spectrum = self.magnitude_spectrum(n_fft_seconds, hop_length_seconds)

        return spectrum.get_spectral_skewness(magnitude_spectrum, self.sample_rate)

    def spectral_kurtosis(self, n_fft_seconds=0.04, hop_length_seconds=0.01):
        """Compute spectral kurtosis from magnitude spectrum.

        Args:
            n_fft_seconds (float): Length of the FFT window in seconds.
            hop_length_seconds (float): How much the window shifts for every timestep,
                in seconds.

        Returns:
            np.array, [1, T / hop_length]: Spectral kurtosis of the magnitude spectrum (fourth moment).
        """
        magnitude_spectrum = self.magnitude_spectrum(n_fft_seconds, hop_length_seconds)

        return spectrum.get_spectral_kurtosis(magnitude_spectrum, self.sample_rate)

    def spectral_flatness(self, n_fft_seconds=0.04, hop_length_seconds=0.01):
        """Given an FFT window size and a hop length, uses the librosa feature package to compute the spectral
        flatness of self.waveform. This component is a measure to quantify how "noise-like" a sound is. The closer
        to 1, the closer the sound is to white noise.

        Args:
            n_fft_seconds (float): Length of the FFT window in seconds.
            hop_length_seconds (float): How much the window shifts for every timestep,
                in seconds.

        Returns:
            np.array, [1, T/hop_length]: Spectral flatness vector computed over windows.
        """
        n_fft = numseconds_to_numsamples(n_fft_seconds, self.sample_rate)
        hop_length = numseconds_to_numsamples(hop_length_seconds, self.sample_rate)

        return librosa.feature.spectral_flatness(
            self.waveform, n_fft=n_fft, hop_length=hop_length
        )

    def spectral_rolloff(self, roll_percent=0.85, n_fft_seconds=0.04, hop_length_seconds=0.01):
        """Given an FFT window size and a hop length, uses the librosa component package to compute the spectral
        roll-off of self.waveform. It is the point below which most energy of a signal is contained and is
        useful in distinguishing sounds with different energy distributions.

        Args:
            roll_percent (float): The roll-off percentage:
                 https://essentia.upf.edu/reference/streaming_RollOff.html
            n_fft_seconds (float): Length of the FFT window in seconds.
            hop_length_seconds (float): How much the window shifts for every timestep,
                in seconds.

        Returns:
            np.array, [1, T/hop_length]: Spectral rolloff vector computed over windows.
        """
        n_fft = numseconds_to_numsamples(n_fft_seconds, self.sample_rate)
        hop_length = numseconds_to_numsamples(hop_length_seconds, self.sample_rate)

        return librosa.feature.spectral_rolloff(
            self.waveform, sr=self.sample_rate, n_fft=n_fft, hop_length=hop_length,
            roll_percent=roll_percent,
        )

    def loudness(self):
        """Compute the loudness of self.waveform using the pyloudnorm package.
        See https://github.com/csteinmetz1/pyloudnorm for more details on potential
        arguments to the functions below.

        Returns:
            float: The loudness of self.waveform
        """
        return get_loudness(self.waveform, self.sample_rate)

    def loudness_slidingwindow(self, frame_length_seconds=1, hop_length_seconds=0.25):
        """Compute the loudness of self.waveform over time. See self.loudness for
        more details.

        Args:
            frame_length_seconds (float): Length of the sliding window in seconds.
            hop_length_seconds (float): How much the sliding window moves by

        Returns:
            [1, T / hop_length]: The loudness on frames of self.waveform
        """
        try:
            return get_loudness_slidingwindow(
                self.waveform, self.sample_rate, frame_length_seconds, hop_length_seconds
            )
        except ValueError:
            raise ValueError(
                "Frames for loudness computation are too short. Consider decreasing the frame length."
            )

    def shannon_entropy(self):
        """Compute the Shannon entropy of self.waveform,
        as per https://ijssst.info/Vol-16/No-4/data/8258a127.pdf

        Returns:
            float: Shannon entropy of the waveform.
        """
        return get_shannon_entropy(self.waveform)

    def shannon_entropy_slidingwindow(self, frame_length_seconds=0.04, hop_length_seconds=0.01):
        """Compute the Shannon entropy of subblocks of a waveform into a newly created time series,
        as per https://ijssst.info/Vol-16/No-4/data/8258a127.pdf

        Args:
            frame_length_seconds (float): Length of the sliding window, in seconds.
            hop_length_seconds (float): How much the window shifts for every timestep,
                in seconds.

        Returns:
            np.array, [1, T / hop_length]: Shannon entropy for each frame
        """
        return get_shannon_entropy_slidingwindow(
            self.waveform, self.sample_rate, frame_length_seconds, hop_length_seconds
        )

    def zerocrossing(self):
        """Compute the zero crossing rate on self.waveform and return it as per
        https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0162128&type=printable
        Note: can also compute zero crossing rate as a time series -- see librosa.feature.zero_crossing_rate,
        and self.get_zcr_sequence.

        Returns:
            dictionary: Keys "num_zerocrossings" and "rate" mapping to: zerocrossing["num_zerocrossings"]:
            number of zero crossings in self.waveform zerocrossing["rate"]: number of zero crossings 
            divided by number of samples.
        """
        num_zerocrossings = librosa.core.zero_crossings(self.waveform).sum()
        rate = num_zerocrossings / self.waveform.shape[0]
        return {"num_zerocrossings": num_zerocrossings, "zerocrossing_rate": rate}

    def zerocrossing_slidingwindow(self, frame_length_seconds=0.04, hop_length_seconds=0.01):
        """Compute the zero crossing rate sequence on self.waveform and return it. This is now a sequence where every entry is
        computed on frame_length samples. There is a sliding window of length hop_length.

        Args:
            frame_length_seconds (float): Length of the sliding window, in seconds.
            hop_length_seconds (float): How much the window shifts for every timestep,
                in seconds.

        Returns:
            np.array, [1, T / hop_length]: Fraction of zero crossings for each frame.
        """

        frame_length = numseconds_to_numsamples(frame_length_seconds, self.sample_rate)
        hop_length = numseconds_to_numsamples(hop_length_seconds, self.sample_rate)

        return librosa.feature.zero_crossing_rate(
            self.waveform, frame_length=frame_length, hop_length=hop_length
        )

    def rms(self, frame_length_seconds=0.04, hop_length_seconds=0.01):
        """Get the root mean square value for each frame, with a specific frame length and hop length. This used
        to be called RMSE, or root mean square energy in the jargon?

        Args:
            frame_length_seconds (float): Length of the sliding window, in seconds.
            hop_length_seconds (float): How much the window shifts for every timestep,
                in seconds.

        Returns:
            np.array, [1, T / hop_length]: RMS value for each frame.
            """
        frame_length = numseconds_to_numsamples(frame_length_seconds, self.sample_rate)
        hop_length = numseconds_to_numsamples(hop_length_seconds, self.sample_rate)

        return librosa.feature.rms(self.waveform, frame_length=frame_length, hop_length=hop_length)

    def intensity(self, frame_length_seconds=0.04, hop_length_seconds=0.01):
        """Get a value proportional to the intensity for each frame, with a specific frame length and hop length.
        Note that the intensity is proportional to the RMS amplitude squared.

        Args:
            frame_length_seconds (float): Length of the sliding window, in seconds.
            hop_length_seconds (float): How much the window shifts for every timestep,
                in seconds.

        Returns:
            np.array, [1, T / hop_length]: Proportional intensity value for each frame.
        """
        return self.rms(
            frame_length_seconds=frame_length_seconds,
            hop_length_seconds=hop_length_seconds
        ) ** 2

    def crest_factor(self, frame_length_seconds=0.04, hop_length_seconds=0.01):
        """Get the crest factor of this waveform, on sliding windows. This value measures the local intensity
        of peaks in a waveform. Implemented as per: https://en.wikipedia.org/wiki/Crest_factor

        Args:
            frame_length_seconds (float): Length of the sliding window, in seconds.
            hop_length_seconds (float): How much the window shifts for every timestep,
                in seconds.

        Returns:
            np.array, [1, T / hop_length]: Crest factor for each frame.
        """
        rms_array = self.rms(
            frame_length_seconds=frame_length_seconds, hop_length_seconds=hop_length_seconds
        )

        return get_crest_factor(
            self.waveform, self.sample_rate, rms_array, frame_length_seconds=frame_length_seconds,
            hop_length_seconds=hop_length_seconds,
        )

    def f0_contour(self, hop_length_seconds=0.01, method='swipe', f0_min=60, f0_max=300):
        """Compute the F0 contour using PYSPTK: https://github.com/r9y9/pysptk/.

        Args:
            hop_length_seconds (float): Hop size argument in pysptk. Corresponds to hopsize
                in the window sliding of the computation of f0. This is in seconds and gets
                converted.
            method (str): One of 'swipe' or 'rapt'. Define which method to use for f0
                calculation. See https://github.com/r9y9/pysptk
            f0_min (float): minimum acceptable f0.
            f0_max (float): maximum acceptable f0.

        Returns:
            np.array, [1, t1]: F0 contour of self.waveform. Contains unvoiced
                frames.
        """
        return get_f0(
            self.waveform, self.sample_rate, hop_length_seconds=hop_length_seconds, method=method,
            f0_min=f0_min, f0_max=f0_max,
        )["contour"]

    def f0_statistics(self, hop_length_seconds=0.01, method='swipe'):
        """Compute the F0 mean and standard deviation of self.waveform. Note that we cannot
        simply rely on using statistics applied to the f0_contour since we do not want to
        include the zeros in the mean and standard deviation calculations.

        Args:
            hop_length_seconds (float): Hop size argument in pysptk. Corresponds to hopsize
                in the window sliding of the computation of f0. This is in seconds and gets
                converted.
            method (str): One of 'swipe' or 'rapt'. Define which method to use for f0
                calculation. See https://github.com/r9y9/pysptk

        Returns:
            dict: Dictionary mapping: "mean": f0 mean of self.waveform. 
                "std": f0 standard deviation of self.waveform.
        """
        f0_dict = get_f0(
            self.waveform, self.sample_rate, hop_length_seconds=hop_length_seconds, method=method
        )

        f0_mean, f0_std = f0_dict["mean"], f0_dict["std"]
        return {'f0_mean': f0_mean, 'f0_std': f0_std}

    def ppe(self):
        """Compute pitch period entropy. This is an adaptation of the following Matlab code:
        https://github.com/Mak-Sim/Troparion/blob/5126f434b96e0c1a4a41fa99dd9148f3c959cfac/Perturbation_analysis/pitch_period_entropy.m
        Note that computing the PPE relies on the existence of voiced portions in the F0 trajectory.

        Returns:
            float: The pitch period entropy, as per http://www.maxlittle.net/students/thesis_tsanas.pdf
        """

        f0_dict = get_f0(self.waveform, self.sample_rate)

        if not np.isnan(f0_dict["mean"]):
            f_min = f0_dict["mean"] / np.sqrt(2)
        else:
            raise ValueError("F0 mean is NaN. Check that the waveform has voiced portions.")

        if len(f0_dict["values"]) > 0:
            rat_f0 = f0_dict["values"] / f_min
        else:
            raise ValueError("F0 does not contain any voiced portions")

        return get_ppe(rat_f0)

    def jitters(self, p_floor=0.0001, p_ceil=0.02, max_p_factor=1.3):
        """Compute the jitters mathematically, according to certain conditions
        given by p_floor, p_ceil and max_p_factor. See jitters.py for more details.

        Args:
            p_floor (float): Minimum acceptable period.
            p_ceil (float): Maximum acceptable period.
            max_p_factor (float): value to use for the period factor principle

        Returns:
            dict: dictionary mapping strings to floats, with keys "localJitter",
            "localabsoluteJitter", "rapJitter", "ppq5Jitter", "ddpJitter"
        """
        jitters_dict = jitters.get_jitters(
            self.f0_contour()[0], p_floor=p_floor,
            p_ceil=p_ceil, max_p_factor=max_p_factor,
        )
        return jitters_dict

    def shimmers(self, max_a_factor=1.6, p_floor=0.0001, p_ceil=0.02, max_p_factor=1.3):
        """Compute the shimmers mathematically, according to certain conditions
        given by max_a_factor, p_floor, p_ceil and max_p_factor.
        See shimmers.py for more details.

        Args:
            max_a_factor (float): Value to use for amplitude factor principle
            p_floor (float): Minimum acceptable period.
            p_ceil (float): Maximum acceptable period.
            max_p_factor (float): value to use for the period factor principle

        Returns:
            dict: Dictionary mapping strings to floats, with keys "localShimmer",
                "localdbShimmer", "apq3Shimmer", "apq5Shimmer", "apq11Shimmer"
        """
        shimmers_dict = shimmers.get_shimmers(
            self.waveform, self.sample_rate, self.f0_contour()[0], max_a_factor=max_a_factor,
            p_floor=p_floor, p_ceil=p_ceil, max_p_factor=max_p_factor,
        )
        return shimmers_dict

    def hnr(self):
        """See https://www.ncbi.nlm.nih.gov/pubmed/12512635 for more thorough description
        of why HNR is important in the scope of healthcare.

        Returns:
            float: The harmonics to noise ratio computed on self.waveform.
        """
        return hnr.get_harmonics_to_noise_ratio(self.waveform, self.sample_rate)

    def dfa(self, window_lengths=[64, 128, 256, 512, 1024, 2048, 4096]):
        """See Tsanas et al, 2011:
        Novel speech signal processing algorithms for high-accuracy classification of Parkinson‟s disease
        Detrended Fluctuation Analysis

        Args:
            window_lengths (list of int > 0): List of L to use in DFA computation.
                See dfa.py for more details.

        Returns:
            float: The detrended fluctuation analysis alpha value.
        """
        return dfa.get_dfa(self.waveform, window_lengths)

    def lpc(self, order=4, return_np_array=False):
        """This uses the librosa backend to get the Linear Prediction Coefficients via Burg's
        method. See librosa.core.lpc for more details.

        Args:
            order (int > 0): Order of the linear filter
            return_np_array (bool): If False, returns a dictionary. Otherwise a
                numpy array.

        Returns:
            dict or np.array, [order + 1, ]: Dictionary mapping 'LPC_{i}' to the i'th lpc coefficient,
            for i = 0...order. Or: LP prediction error coefficients (np array case)
        """
        lpcs = librosa.core.lpc(self.waveform, order=order)
        if return_np_array:
            return lpcs
        return {f'LPC_{i}': lpc for i, lpc in enumerate(lpcs)}

    def lsf(self, order=4, return_np_array=False):
        """Compute the LPC coefficients, then convert them to LSP frequencies. The conversion is
        done using https://github.com/cokelaer/spectrum/blob/master/src/spectrum/linear_prediction.py

        Args:
            order (int > 0): Order of the linear filter for LPC calculation
            return_np_array (bool): If False, returns a dictionary. Otherwise a
                numpy array.

        Returns:
            dict or np.array, [order, ]: Dictionary mapping 'LPC_{i}' to the
                i'th lpc coefficient, for i = 0...order. Or LSP frequencies (np array case).
        """
        lpc_poly = self.lpc(order, return_np_array=True)
        lsfs = np.array(lpc_to_lsf(lpc_poly))
        if return_np_array:
            return lsfs
        return {f'LSF_{i}': lsf for i, lsf in enumerate(lsfs)}

    def formants(self):
        """Estimate the first four formant frequencies using LPC (see formants.py)

        Returns:
            dict: Dictionary mapping {'f1', 'f2', 'f3', 'f4'} to
            corresponding {first, second, third, fourth} formant frequency.
        """
        formants_dict = formants.get_formants(self.waveform, self.sample_rate)
        return formants_dict

    def formants_slidingwindow(self, frame_length_seconds=0.04, hop_length_seconds=0.01):
        """Estimate the first four formant frequencies using LPC (see formants.py) and
        apply the metric_slidingwindow decorator.

        Args:
            frame_length_seconds (float): Length of the sliding window, in seconds.
            hop_length_seconds (float): How much the window shifts for every timestep,
                in seconds.

        Returns:
            np.array, [4, T / hop_length]: Time series of the first four formant frequencies
                computed on windows of length frame_length_seconds, with sliding window of
                hop_length_seconds.
        """
        try:
            formants_nparray = np.concatenate([
                formants.get_formants_slidingwindow(
                    self.waveform, self.sample_rate, f, frame_length_seconds=frame_length_seconds,
                    hop_length_seconds=hop_length_seconds,
                ) for f in ["f1", "f2", "f3", "f4"]
            ])
            return formants_nparray
        except FloatingPointError:
            raise ValueError("Input seems to be ill conditioned.")

    def kurtosis_slidingwindow(self, frame_length_seconds=0.04, hop_length_seconds=0.01):
        """Computes the kurtosis on frames of the waveform with a sliding
        window

        Args:
            frame_length_seconds (float): Length of the sliding window, in seconds.
            hop_length_seconds (float): How much the window shifts for every timestep,
                in seconds.

        Returns:
            np.array, [1, T / hop_length]: Kurtosis on each sliding window.
        """
        return get_kurtosis_slidingwindow(
            self.waveform, self.sample_rate, frame_length_seconds, hop_length_seconds
        )

    def log_energy(self):
        """Compute the log energy of self.waveform as per Abeyrante et al. 2013.

        Returns:
            float: The log energy of self.waveform, computed as per the paper above.
        """
        return get_log_energy(self.waveform)

    def log_energy_slidingwindow(self, frame_length_seconds=0.04, hop_length_seconds=0.01):
        """Computes the log energy on frames of the waveform with a sliding
        window

        Args:
            frame_length_seconds (float): Length of the sliding window, in seconds.
            hop_length_seconds (float): How much the window shifts for every timestep,
                in seconds.
        Returns:
            np.array, [1, T / hop_length]: Log energy on each sliding window.
        """
        return get_log_energy_slidingwindow(
            self.waveform, self.sample_rate, frame_length_seconds, hop_length_seconds
        )
