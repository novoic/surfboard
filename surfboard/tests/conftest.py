import pytest
from pandas import np

from surfboard import sound
from surfboard.utils import example_audio_file


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