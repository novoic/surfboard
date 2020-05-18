#!/usr/bin/env python
"""This file contains functions to compute features."""

import numpy as np
import pandas as pd

from tqdm import tqdm

from .sound import Waveform
from .statistics import Barrel


def load_waveforms_from_paths(paths, sample_rate):
    """Loads waveforms from paths using multiprocessing"""
    progress_bar = tqdm(paths, desc='Loading waveforms...')
    return [Waveform(path=p, sample_rate=sample_rate) for p in progress_bar]


def extract_features_from_paths(paths, components_list, statistics_list=None, sample_rate=44100):
    """Function which loads waveforms, computes the components and statistics and returns them,
    without the need to store the waveforms in memory. This is to minimize the memory footprint
    when running over multiple files.

    Args:
        paths (list of str): .wav to compute
        components_list (list of str/dict): This is a list of the methods which
            should be applied to all the waveform objects in waveforms. If a dict,
            this also contains arguments to the sound.Waveform methods.
        statistics_list (list of str): This is a list of the methods which
            should be applied to all the time-dependent features computed
            from the waveforms.
        sample_rate (int > 0): sampling rate to load the waveforms

    Returns:
        pandas DataFrame: pandas dataframe where every row corresponds
            to features extracted for one of the waveforms and columns
            represent individual features.
    """
    output_feats = []
    paths = tqdm(paths, desc='Extracting features from paths...')

    for path in paths:
        wave = Waveform(path=path, sample_rate=sample_rate)
        output_feats.append(
            extract_features_from_waveform(
                components_list, statistics_list, wave
            )
        )

    return pd.DataFrame(output_feats)


def extract_features_from_waveform(components_list, statistics_list, waveform):
    """Given one waveform, a list of components and statistics, extract the
    features from the waveform.

    Args:
        components_list (list of str or dict): This is a list of the methods which
            should be applied to all the waveform objects in waveforms. If a dict,
            this also contains arguments to the sound.Waveform methods.
        statistics_list (list of str): This is a list of the methods which
            should be applied to all the "time-dependent" components computed
            from the waveforms.
        waveform (Waveform): the waveform object to extract components from.

    Returns:
        dict: Dictionary mapping names to numerical components extracted
            for this waveform.
    """
    feats_this_waveform = {}
    try:
        # Compute components with surfboard.
        components = waveform.compute_components(components_list)

        # Loop over computed components to either prepare for output, or to apply statistics.
        for component_name in components:
            # Case of a dictionary: unpack dictionary and merge with existing set of components.
            if isinstance(components[component_name], dict) and statistics_list is not None:
                feats_this_waveform = {
                    **feats_this_waveform,
                    **components[component_name]
                }

            # Case of a float -- simply add that as a single value to the dictionary.
            # Or: case of a np array when statistics list is None. In order to be able to obtain
            # the numpy array from the pandas DataFrame, we must pass the np array as a list.
            elif isinstance(components[component_name], float) or (
                isinstance(components[component_name], np.ndarray) and statistics_list is None
            ):
                feats_this_waveform[component_name] = components[component_name]

            # Case of a np.array (the component is a time series). Apply Barrel.
            elif isinstance(components[component_name], np.ndarray) and statistics_list is not None:
                barrel = Barrel(components[component_name])
                function_outputs = barrel.compute_statistics(statistics_list)
                # Merge dictionaries...
                feats_this_waveform = {
                    **feats_this_waveform,
                    **{"{}_{}".format(component_name, fun_name): v for fun_name, v in function_outputs.items()}
                }

    except Exception as extraction_exception:
        print(f'Found exception "{extraction_exception}"... Skipping...')
        return {}

    except:
        print('Unknow error. Skipping')
        return {}

    # Return an empty dict in the case of None.
    feats_this_waveform = feats_this_waveform if feats_this_waveform is not None else {}
    return feats_this_waveform


def extract_features(waveforms, components_list, statistics_list=None):
    """This is an important function. Given a list of Waveform objects, a list of
    Waveform methods in the form of strings and a list of Barrel methods in the
    form of strings, compute the time-independent features resulting. This function
    does multiprocessing.

    Args:
        waveforms (list of Waveform): This is a list of waveform objects
        components_list (list of str/dict): This is a list of the methods which
            should be applied to all the waveform objects in waveforms. If a dict,
            this also contains arguments to the sound.Waveform methods.
        statistics_list (list of str): This is a list of the methods which
            should be applied to all the time-dependent features computed
            from the waveforms.

    Returns:
        pandas DataFrame: pandas dataframe where every row corresponds
            to features extracted for one of the waveforms and columns
            represent individual features.
    """
    output_feats = []
    waveforms = tqdm(waveforms, desc='Extracting features...')

    for wave in waveforms:
        output_feats.append(
            extract_features_from_waveform(
                components_list, statistics_list, wave
            )
        )

    return pd.DataFrame(output_feats)
