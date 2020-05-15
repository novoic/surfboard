import pytest

import numpy as np

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


def test_constructor_signal(flat_waveform):
    """Test Waveform constructor from np array"""
    sound = flat_waveform.waveform
    assert isinstance(sound, np.ndarray)


def test_constructor_path(waveform):
    """Test Waveform constructor from a path"""
    sound = waveform.waveform
    assert isinstance(sound, np.ndarray)


def test_mfcc(waveform):
    """Test MFCCs"""
    mfcc = waveform.mfcc()
    assert isinstance(mfcc, np.ndarray)


def test_log_melspec(waveform):
    """Test Log Mel Spectrogram"""
    melspec = waveform.log_melspec()
    assert isinstance(melspec, np.ndarray)


def test_magnitude_spectrum(waveform):
    """Test Magnitude Spectrum"""
    S = waveform.magnitude_spectrum()
    assert isinstance(S, np.ndarray)


def test_bark_spectrogram(waveform):
    """Test Bark Spectrogram"""
    S = waveform.bark_spectrogram()
    assert isinstance(S, np.ndarray)


def test_morlet_cwt(flat_waveform):
    """Test Morlet Continuous Wavelet Transform
    Flat waveform because too long otherwise.
    """
    cwt = flat_waveform.morlet_cwt()
    assert isinstance(cwt, np.ndarray)


def test_chroma_stft(waveform):
    """Test Chromagram from STFT"""
    chroma_stft = waveform.chroma_stft()
    assert isinstance(chroma_stft, np.ndarray)


def test_chroma_cqt(waveform):
    """Test Chromagram from CQT"""
    chroma_cqt = waveform.chroma_cqt()
    assert isinstance(chroma_cqt, np.ndarray)


def test_chroma_cens(waveform):
    """Test Chroma CENS"""
    chroma_cens = waveform.chroma_cens()
    assert isinstance(chroma_cens, np.ndarray)


def test_spectral_slope(waveform):
    """Test spectral slope"""
    spectral_slope = waveform.spectral_slope()
    assert isinstance(spectral_slope, np.ndarray)


def test_spectral_flux(waveform):
    """Test spectral flux"""
    spectral_flux = waveform.spectral_flux()
    assert isinstance(spectral_flux, np.ndarray)


def test_spectral_entropy(waveform):
    """Test spectral entropy"""
    spectral_entropy = waveform.spectral_entropy()
    assert isinstance(spectral_entropy, np.ndarray)


def test_spectral_centroid(waveform):
    """Test spectral centroid"""
    spectral_centroid = waveform.spectral_centroid()
    assert isinstance(spectral_centroid, np.ndarray)


def test_spectral_spread(waveform):
    """Test spectral spread"""
    spectral_spread = waveform.spectral_spread()
    assert isinstance(spectral_spread, np.ndarray)


def test_spectral_skewness(waveform):
    """Test spectral skewness"""
    spectral_skewness = waveform.spectral_skewness()
    assert isinstance(spectral_skewness, np.ndarray)


def test_spectral_kurtosis(waveform):
    """Test spectral kurtosis"""
    spectral_kurtosis = waveform.spectral_kurtosis()
    assert isinstance(spectral_kurtosis, np.ndarray)


def test_spectral_flatness(waveform):
    """Test spectral flatness"""
    spectral_flatness = waveform.spectral_flatness()
    assert isinstance(spectral_flatness, np.ndarray)


def test_spectral_rolloff(waveform):
    """Test spectral rolloff"""
    spectral_rolloff = waveform.spectral_rolloff()
    assert isinstance(spectral_rolloff, np.ndarray)


def test_loudness(waveform):
    """Test loudness"""
    loudness = waveform.loudness()
    assert isinstance(loudness, float)


def test_loudness_slidingwindow(waveform):
    """Test loudness over sliding windows"""
    loudness_slidingwindow = waveform.loudness_slidingwindow()
    assert isinstance(loudness_slidingwindow, np.ndarray)


def test_shannon_entropy(waveform):
    """Test Shannon entropy"""
    shannon_entropy = waveform.shannon_entropy()
    assert isinstance(shannon_entropy, float)


def test_shannon_entropy_slidingwindow(waveform):
    """Test Shannon entropy over sliding windows"""
    shannon_entropy = waveform.shannon_entropy_slidingwindow()
    assert isinstance(shannon_entropy, np.ndarray)


def test_zerocrossing(waveform):
    """Test zero crossing"""
    zerocrossing = waveform.zerocrossing()
    assert isinstance(zerocrossing, dict)


def test_zerocrossing_slidingwindow(waveform):
    """Test zero crossing over sliding windows"""
    zerocrossing_slidingwindow = waveform.zerocrossing_slidingwindow()
    assert isinstance(zerocrossing_slidingwindow, np.ndarray)


def test_rms(waveform):
    """Test energy"""
    rms = waveform.rms()
    assert isinstance(rms, np.ndarray)


def test_intensity(waveform):
    """Test intensity"""
    intensity = waveform.intensity()
    assert isinstance(intensity, np.ndarray)


def test_crest_factor(waveform):
    """Test crest factor"""
    crest_factor = waveform.crest_factor()
    assert isinstance(crest_factor, np.ndarray)


def test_f0_swipe(waveform):
    """Test f0 with SWIPE method"""
    f0 = waveform.f0_contour(method='swipe')
    assert isinstance(f0, np.ndarray)


def test_f0_rapt(waveform):
    """Test f0 with RAPT method"""
    f0 = waveform.f0_contour(method='rapt')
    assert isinstance(f0, np.ndarray)


def test_f0_statistics(waveform):
    """Test f0 statistics"""
    f0_statistics = waveform.f0_statistics(method='rapt')
    assert isinstance(f0_statistics, dict)


def test_ppe(waveform):
    """Test pitch period entropy"""
    try:
        ppe = waveform.ppe()
    except ValueError:
        return
    assert isinstance(ppe, float)


def test_jitters(waveform):
    """Test jitters"""
    jitters = waveform.jitters()
    assert isinstance(jitters, dict)


def test_shimmers(waveform):
    """Test shimmers"""
    shimmers = waveform.shimmers()
    assert isinstance(shimmers, dict)


def test_hnr(waveform):
    """Test harmonics to noise ratio"""
    hnr = waveform.hnr()
    assert isinstance(hnr, float)


def test_dfa(waveform):
    """Test detrended fluctuation analysis"""
    dfa = waveform.dfa()
    assert isinstance(dfa, float)


def test_lpc(waveform):
    """Test Linear Prediction Coefficients"""
    lpc = waveform.lpc(order=200)
    assert isinstance(lpc, dict)


def test_lsf(waveform):
    """Test Linear Spectral Frequencies"""
    lsf = waveform.lsf(order=200)
    assert isinstance(lsf, dict)


def test_formants(waveform):
    """Test Formants"""
    formants = waveform.formants()
    assert isinstance(formants, dict)


def test_formants_slidingwindow(waveform):
    """Test Formants computes over sliding windows"""
    try:
        formants_slidingwindow = waveform.formants_slidingwindow()
    except ValueError:
        return
    assert isinstance(formants_slidingwindow, np.ndarray)


def test_kurtosis_slidingwindow(waveform):
    """Test kurtosis computed over sliding windows"""
    kurtosis = waveform.kurtosis_slidingwindow()
    assert isinstance(kurtosis, np.ndarray)


def test_log_energy(waveform):
    """Test log energy"""
    log_energy = waveform.log_energy()
    assert isinstance(log_energy, float)


def test_log_energy_slidingwindow(waveform):
    """Test log energy computed over sliding windows"""
    log_energy = waveform.log_energy_slidingwindow()
    assert isinstance(log_energy, np.ndarray)
