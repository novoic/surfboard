#!/usr/bin/env python
"""This file contains a variety of helper functions for the surfboard package."""

import os

import numpy as np

from scipy.signal import (
    deconvolve,
)


def metric_slidingwindow(frame_length, hop_length, truncate_end=False):
    """We use this decorator to decorate functions which take a sequence
    as an input and return a metric (float). For example the sum of a sequence.
    This decorator will enable us to quickly compute the metrics over a sliding
    window. Note the existence of the implicit decorator below which allows us
    to have arguments to the decorator.

    Args:
        frame_length (int): The length of the sliding window
        hop_length (int): How much to slide the window every time
        truncate_end (bool): whether to drop frames which are shorter than
            frame_length (the end frames, typically)

    Returns:
        function: The function which computes the metric over sliding
            windows.
    """

    def implicit_decorator(func):
        def wrapper(*args):
            sequence = args[0]
            output_list = []
            for i in range(int(np.ceil(sequence.shape[0] / hop_length))):
                subblock = sequence[i * hop_length: i * hop_length + frame_length]
                if len(subblock) < frame_length and truncate_end:
                    continue

                # Change the first argument (the sequence). Keep the rest of the arguments.
                new_args = (subblock, *args[1:])
                value = func(*new_args)
                output_list.append(value)
            out = np.expand_dims(np.array(output_list), 0)
            return out

        return wrapper

    return implicit_decorator


def numseconds_to_numsamples(numseconds, sample_rate):
    """Convert a number of seconds a sample rate to the number of samples for n_fft,
    frame_length and hop_length computation. Find the closest power of 2 for efficient
    computations.

    Args:
        numseconds (float): number of seconds that we want to convert
        sample_rate (int): how many samples per second

    Return:
        int: closest power of 2 to int(numseconds * sample_rate)
    """
    candidate = int(numseconds * sample_rate)
    log2 = np.log2(candidate)
    out_value = int(2 ** np.round(log2))

    assert out_value != 0, "The inputs given gave an output value of 0. This is not acceptable."

    return out_value


def max_peak_amplitude(signal):
    """Returns the maximum absolute value of a signal.

    Args:
        np.array [T, ]: a waveform

    Returns:
        float: the maximum amplitude of this waveform, in absolute value
    """
    return np.max(np.abs(signal))


def peak_amplitude_slidingwindow(signal, sample_rate, frame_length_seconds=0.04, hop_length_seconds=0.01):
    """Apply the metric_slidingwindow decorator to the the peak amplitude computation defined above,
    effectively computing frequency from fft over sliding windows.

    Args:
        signal (np.array [T,]): waveform over which to compute.
        sample_rate (int): number of samples per second in the waveform
        frame_length_seconds (float): how many seconds in one frame. This
            value is defined in seconds instead of number of samples.
        hop_length_seconds (float): how many seconds frames shift each step.
            This value is defined in seconds instead of number of samples.

    Returns:
        np.array, [1, T / hop_length]: peak amplitude on each window.
    """
    frame_length = numseconds_to_numsamples(frame_length_seconds, sample_rate)
    hop_length = numseconds_to_numsamples(hop_length_seconds, sample_rate)

    @metric_slidingwindow(frame_length=frame_length, hop_length=hop_length)
    def new_max_peak_amplitude(signal):
        return max_peak_amplitude(signal)

    return new_max_peak_amplitude(signal)


def shifted_sequence(sequence, num_sequences):
    """Given a sequence (say a list) and an integer, returns a zipped iterator
    of sequence[:-num_sequences + 1], sequence[1:-num_sequences + 2], etc.

    Args:
        sequence (list or other iteratable): the sequence over which to iterate
            in various orders
        num_sequences (int): the number of sequences over which we iterate.
            Also the number of elements which come out of the output at each call.

    Returns:
        iterator: zipped shifted sequences.
    """
    return zip(
        *(
            [list(sequence[i: -num_sequences + 1 + i]) for i in range(
                num_sequences - 1)] + [sequence[num_sequences - 1:]]
        )
    )


def lpc_to_lsf(lpc_polynomial):
    """This code is inspired by the following:
    https://uk.mathworks.com/help/dsp/ref/lpctolsflspconversion.html

    Args:
        lpc_polynomial (list): length n + 1 list of lpc coefficients. Requirements
            is that the polynomial is ordered so that lpc_polynomial[0] == 1

    Returns:
        list: length n list of line spectral frequencies.
    """
    lpc_polynomial = np.array(lpc_polynomial)

    assert lpc_polynomial[0] == 1, \
        'First value in the polynomial must be 1. Considering normalizing.'

    assert max(np.abs(np.roots(lpc_polynomial))) <= 1.0, \
        'The polynomial must have all roots inside of the unit circle.'

    lhs = np.concatenate((lpc_polynomial, np.array([0])))
    rhs = lhs[-1::-1]
    diff_filter = lhs - rhs
    sum_filter = lhs + rhs

    poly_1 = deconvolve(diff_filter, [1, 0, -1])[0] if (len(lpc_polynomial) - 1) % 2 else deconvolve(diff_filter, [1, -1])[0]
    poly_2 = sum_filter if (len(lpc_polynomial) - 1) % 2 else deconvolve(sum_filter, [1, 1])[0]

    roots_poly1 = np.roots(poly_1)
    roots_poly2 = np.roots(poly_2)

    angles_poly1 = np.angle(roots_poly1[1::2])
    angles_poly2 = np.angle(roots_poly2[1::2])

    return sorted(
        np.concatenate((-angles_poly1, -angles_poly2))
    )


def parse_component(component):
    """Parse the component coming from the .yaml file.

    Args:
        component (str or dict): Can be either a str, or a dictionary.
            Comes from the .yaml config file. If it is a string,
            simply return, since its the component name without arguments.
            Otherwise, parse.

    Returns:
        tuple: tuple containing:
            str: name of the method to be called from sound.Waveform
            dict: arguments to be unpacked. None if no arguments to
                compute.
    """
    if isinstance(component, str):
        return component, None
    elif isinstance(component, dict):
        component_name = list(component.keys())[0]
        # component[component_name] is a dictionary of arguments.
        arguments = component[component_name]
        return component_name, arguments
    else:
        raise ValueError("Argument to the parse_component function must be str or dict.")


def example_audio_file(which_file):
    """Returns the path to one of sustained_a, sustained_o or sustained_e
    included with the Surfboard package.

    Args:
        which_file (str): One of 'a', 'o' or 'e'

    Returns:
        str: The path to the chosen file.
    """
    assert which_file in ['a', 'o', 'e'], 'Input must be one of: "a", "o", "e"'
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        f'../example_audio_files/sustained_{which_file}.wav'
    )


class YamlFileException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message
