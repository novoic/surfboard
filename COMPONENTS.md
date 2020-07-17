Component Name | Implementation | Component Code | Method Arguments
------------ | ------------- | ------------ | ------------  
MFCCs | LibROSA | `mfcc` |  `n_mfcc=13`, `n_fft_seconds=0.04`, `hop_length_seconds=0.01`
Log mel spectrogram | LibROSA | `log_melspec` | `n_mels=128`, `n_fft_seconds=0.04`, `hop_length_seconds=0.01`
Morlet continuous wavelet transform | SciPy | `morlet_cwt` | `widths=None`
Bark spectrogram | Ours | `bark_spectrogram` | `n_fft_seconds=0.04`, `hop_length_seconds=0.01`
Magnitude spectrum | LibROSA | `magnitude_spectrum` | `n_fft_seconds=0.04`, `hop_length_seconds=0.01`
Chromagram with STFT | LibROSA | `chroma_stft` | `n_fft_seconds=0.04`, `hop_length_seconds=0.01`, `n_chroma=12`
Chromagram with CQT | LibROSA | `chroma_cqt` | `hop_length_seconds=0.01`, `n_chroma=12`
Chroma CENS | LibROSA | `chroma_cens` | `hop_length_seconds=0.01`, `n_chroma=12`
Spectral slope | pyACA | `spectral_slope` | `n_fft_seconds=0.04`, `hop_length_seconds=0.01`
Spectral flux | pyACA | `spectral_flux` | `n_fft_seconds=0.04`, `hop_length_seconds=0.01`
Spectral entropy | Ours | `spectral_entropy` | `n_fft_seconds=0.04`, `hop_length_seconds=0.01`
Spectral centroid | pyACA | `spectral_centroid` | `n_fft_seconds=0.04`, `hop_length_seconds=0.01`
Spectral spread | pyACA | `spectral_spread` | `n_fft_seconds=0.04`, `hop_length_seconds=0.01`
Spectral skewness | pyACA | `spectral_skewness` | `n_fft_seconds=0.04`, `hop_length_seconds=0.01`
Spectral kurtosis | pyACA | `spectral_kurtosis` | `n_fft_seconds=0.04`, `hop_length_seconds=0.01`
Spectral flatness | LibROSA | `spectral_skewness` | `n_fft_seconds=0.04`, `hop_length_seconds=0.01`
Spectral rolloff | LibROSA | `spectral_skewness` | `n_fft_seconds=0.04`, `hop_length_seconds=0.01`
F0 contour | pysptk | `f0_contour` | `hop_length_seconds=0.01`, `method='swipe'`
F0 statistics (mean, standard deviation) | pysptk | `f0_statistics` | `hop_length_seconds=0.01`, `method='swipe'`
Intensity | Ours | `intensity` | `frame_length_seconds=0.04`, `hop_length_seconds=0.01`
Energy | LibROSA | `rms` | `frame_length_seconds=0.04`, `hop_length_seconds=0.01`
Log-energy | Ours | `log_energy` | None
Sliding window log-energy | Ours | `log_energy_slidingwindow` | `frame_length_seconds=0.04`, `hop_length_seconds=0.01`
Zero-crossing | LibROSA | `zerocrossing` | None
Sliding-window zero-crossing | LibROSA | `zerocrossing_slidingwindow` | `frame_length_seconds=0.04`, `hop_length_seconds=0.01`
Loudness | pyloudnorm | `loudness` | None
Sliding-window loudness | pyloudnorm | `loudness_slidingwindow` | `frame_length_seconds=1`, `hop_length_seconds=0.25`
Crest factor | Ours | `crest_factor` | `frame_length_seconds=0.04`, `hop_length_seconds=0.01`
Pitch period entropy | Ours | `ppe` | None
Jitters | Ours | `jitters` | `p_floor=0.0001`, `p_ceil=0.02`, `max_p_factor=1.3`
Shimmers | Ours | `shimmers` | `max_a_factor=1.6`, `p_floor=0.0001`, `p_ceil=0.02`, `max_p_factor=1.3`
Harmonics-to-noise ratio | Ours | `hnr` | None
Detrended fluctuation analysis | Ours | `dfa` | `window_lengths=[64, 128, 256, 512, 1024, 2048, 4096]`
Linear spectral coefficients | LibROSA | `lpc` | `order=4`, `return_np_array=False`
Linear spectral frequencies | pyspectrum | `lsf` | `order=4`, `return_np_array=False`
Formants (F1, F2, F3, F4) | Ours | `formants` | None
Sliding-window formants (F1, F2, F3, F4) | Ours | `formants_slidingwindow` | `frame_length_seconds=0.04`, `hop_length_seconds=0.01`
Amplitude shannon entropy | Ours | `shannon_entropy` | `frame_length_seconds=0.04`, `hop_length_seconds=0.01`
Sliding-window amplitude shannon entropy | Ours | `shannon_entropy_slidingwindow` | `frame_length_seconds=0.04`, `hop_length_seconds=0.01`
Sliding-window amplitude kurtosis | Ours | `kurtosis_slidingwindow` | `frame_length_seconds=0.04`, `hop_length_seconds=0.01`
