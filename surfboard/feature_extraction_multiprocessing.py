#!/usr/bin/env python
"""This file contains functions to compute features with multiprocessing."""

from multiprocessing import Pool
from functools import partial

import pandas as pd

from tqdm import tqdm

from .feature_extraction import (
    extract_features_from_waveform,
)

from .sound import Waveform


def load_waveform_from_path(sample_rate, path):
    """Helper function to access constructor with Pool

    Args:
        sample_rate (int): The sample rate to load the Waveform object
        path (str): The path to the audio file to load

    Returns:
        Waveform: The loaded Waveform object
    """
    return Waveform(path=path, sample_rate=sample_rate)


def load_waveforms_from_paths(paths, sample_rate, num_proc=1):
    """Loads waveforms from paths using multiprocessing

    Args:
        paths (list of str): A list of paths to audio files
        sample_rate (int): The sample rate to load the audio files
        num_proc (int >= 1): The number of parallel processes to run

    Returns:
        list of Waveform: List of loaded Waveform objects
    """
    assert (num_proc > 0 and isinstance(num_proc, int)), 'The number of parallel \
        processes should be a >= 1 integer.'
    load_helper = partial(load_waveform_from_path, sample_rate)
    with Pool(num_proc) as pool:
        waveforms_iter = tqdm(
            pool.imap(load_helper, paths), total=len(paths), desc='Loading waveforms...'
        )
        # Converting to list runs the iterator.
        output_waveforms = list(waveforms_iter)

    return output_waveforms


def extract_features_from_path(components_list, statistics_list, sample_rate, path):
    """Function which loads a waveform, computes the components and statistics and returns them,
    without the need to store the waveforms in memory. This is to prevent accumulating too
    much memory.

    Args:
        components_list (list of str/dict): This is a list of the methods which
            should be applied to all the waveform objects in waveforms. If a dict,
            this also contains arguments to the sound.Waveform methods.
        statistics_list (list of str): This is a list of the methods which
            should be applied to all the "time-dependent" features computed
            from the waveforms.
        sample_rate (int > 0): sampling rate to load the waveforms
        path (str): path to audio file to extract features from

    Returns:
        dict: Dictionary mapping feature names to values.
    """
    try:
        wave = Waveform(path=path, sample_rate=sample_rate)
        feats = extract_features_from_waveform(components_list, statistics_list, wave)
        return feats
    except Exception as extraction_exception:
        print(f'Found exception "{extraction_exception}". Skipping {path}')
        return {}
    except:
        print(f'Unknown error. Skipping {path}')
        return {}


def extract_features_from_paths(paths, components_list, statistics_list=None, sample_rate=44100, num_proc=1):
    """Function which loads waveforms, computes the features and statistics and returns them,
    without the need to store the waveforms in memory. This is to prevent accumulating too
    much memory.

    Args:
        paths (list of str): .wav to compute
        components_list (list of str or dict): This is a list of the methods which
            should be applied to all the waveform objects in waveforms. If a dict,
            this also contains arguments to the sound.Waveform methods.
        statistics_list (list of str): This is a list of the methods which
            should be applied to all the "time-dependent" features computed
            from the waveforms.
        sample_rate (int > 0): sampling rate to load the waveforms

    Returns:
        pandas DataFrame: pandas dataframe where every row corresponds
            to features extracted for one of the waveforms and columns
            represent individual features.
    """
    extractor_helper = partial(
        extract_features_from_path, components_list, statistics_list, sample_rate
    )
    with Pool(num_proc) as pool:
        output_feats_iter = tqdm(
            pool.imap(extractor_helper, paths), total=len(paths),
            desc='Extracting features from paths...'
        )
        # Converting to list runs the iterator.
        output_feats = list(output_feats_iter)
    output_df = pd.DataFrame(output_feats)
    # Ensure the output DataFrame has the same length as input paths. That way, we can
    # guarantee that the names correspond to the correct rows.
    assert len(output_df) == len(paths), "Output DataFrame does not have same length as \
        input list of paths."
    return output_df


def extract_features(waveforms, components_list, statistics_list=None, num_proc=1):
    """This is an important function. Given a list of Waveform objects, a list of
    Waveform methods in the form of strings and a list of Barrel methods in the
    form of strings, compute the time-independent features resulting. This function
    does multiprocessing.

    Args:
        waveforms (list of Waveform): This is a list of waveform objects
        components_list (list of str or dict): This is a list of the methods which
            should be applied to all the waveform objects in waveforms. If a dict,
            this also contains arguments to the sound.Waveform methods.
        statistics_list (list of str): This is a list of the methods which
            should be applied to all the "time-dependent" features computed
            from the waveforms.
        num_proc (int >= 1): The number of parallel processes to run

    Returns:
        pandas DataFrame: pandas dataframe where every row corresponds
            to features extracted for one of the waveforms and columns
            represent individual features.
    """

    extractor_helper = partial(
        extract_features_from_waveform, components_list, statistics_list
    )

    with Pool(num_proc) as pool:
        output_feats_iter = tqdm(
            pool.imap(extractor_helper, waveforms), total=len(waveforms),
            desc='Extracting features...'
        )
        # Converting to list runs the iterator.
        output_feats = list(output_feats_iter)

    return pd.DataFrame(output_feats)
