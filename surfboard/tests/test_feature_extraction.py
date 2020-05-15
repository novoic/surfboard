import os
import pytest
import yaml

import pandas as pd
import numpy as np

from surfboard import sound
from surfboard.utils import example_audio_file

from surfboard.feature_extraction import (
    extract_features,
    extract_features_from_paths
)


@pytest.fixture
def flat_waveform():
    wave = np.ones((24000,))
    return sound.Waveform(signal=wave, sample_rate=24000)


@pytest.fixture
def waveform():
    filename = example_audio_file('a')
    return sound.Waveform(
        path=filename, sample_rate=24000
    )


def test_extract_features(waveform, flat_waveform):
    """Test the extract features function with and without statistics from the 
    all_features.yaml example config.

    Args:
        waveform (Waveform): The waveform PyTest fixture returning an
            example audio file.
        flat_waveform (Waveform): The flat_waveform PyTest fixture returning
            a flat wave of ones.
    """
    config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../../example_configs/all_features.yaml'
    )
    config = yaml.full_load(open(config_path, 'r'))
    components_list = list(config['components'])
    statistics_list = list(config['statistics'])

    output_without_statistics = extract_features(
        [waveform, flat_waveform], components_list
    )
    # Check correct return type.
    assert isinstance(
        output_without_statistics, pd.DataFrame
    )
    # Check that not all values are NaNs.
    assert not output_without_statistics.isnull().values.all()

    
    output_with_statistics = extract_features(
        [waveform, flat_waveform], components_list, statistics_list
    )
    # Check correct return type.
    assert isinstance(output_with_statistics, pd.DataFrame)
    # Check that not all values are NaNs.
    assert not output_with_statistics.isnull().values.all()


def test_extract_features_from_paths():
    """Test the extract features from paths function with and without
    statistics from the all_features.yaml example config.
    """
    config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../../example_configs/all_features.yaml'
    )
    config = yaml.full_load(open(config_path, 'r'))
    components_list = list(config['components'])
    statistics_list = list(config['statistics'])

    output_without_statistics = extract_features_from_paths(
        [example_audio_file('e')], components_list 
    )
    # Check correct return type.
    assert isinstance(
        output_without_statistics, pd.DataFrame
    )
    # Check that not all values are NaNs.
    assert not output_without_statistics.isnull().values.all()

    output_with_statistics = extract_features_from_paths(
        [example_audio_file('o')], components_list , statistics_list
    )
    # Check correct return type.
    assert isinstance(output_with_statistics, pd.DataFrame)
    # Check that not all values are NaNs.
    assert not output_with_statistics.isnull().values.all()