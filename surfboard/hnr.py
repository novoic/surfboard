"""This function is inspired by the Speech Analysis repository at
https://github.com/brookemosby/Speech_Analysis
"""

import numpy as np
import peakutils as pu


def get_harmonics_to_noise_ratio(
    waveform, sample_rate, min_pitch=75.0, silence_threshold=0.1, periods_per_window=4.5
):
    """Given a waveform, its sample rate, some conditions for voiced and unvoiced
    frames (including min pitch and silence threshold), and a "periods per window"
    argument, compute the harmonics to noise ratio. This is a good measure of
    voice quality and is an important metric in cognitively impaired patients.
    Compute the mean hnr_vector: harmonics to noise ratio.

    Args:
        waveform (np.array, [T, ]): waveform signal
        sample_rate (int > 0): sampling rate of the waveform
        min_pitch (float > 0): minimum acceptable pitch. converts to
            maximum acceptable period.
        silence_threshold (1 >= float >= 0): needs to be in [0, 1]. Below this
            amplitude, does not consider frames.
        periods_per_window (float > 0): 4.5 is best for speech.

    Returns:
        float: Harmonics to noise ratio of the entire considered
            waveform.
    """
    assert min_pitch > 0, "Min pitch needs to be > 0"
    assert 0 <= silence_threshold <= 1, "Silence threshold need to be in [0, 1]"

    hop_length_seconds = periods_per_window / (4.0 * min_pitch)
    window_length_seconds = periods_per_window / min_pitch

    hop_length = int(hop_length_seconds * sample_rate)
    window_length = int(window_length_seconds * sample_rate)

    # Now we need to segment the waveform.
    frames_iterator = range(max(1, int(waveform.shape[0] / hop_length + 0.5)) + 1)
    segmented_waveform = [
        waveform[i * hop_length: i * hop_length + window_length] for i in frames_iterator
    ]

    waveform_peak = max(abs(waveform - waveform.mean()))
    hnr_vector = []

    # Start looping
    for index, chunk in enumerate(segmented_waveform):
        if chunk.shape[0] > 0:
            thischunk_length = chunk.shape[0] / sample_rate
            chunk = chunk - chunk.mean()

            thischunk_peak = np.max(np.abs(chunk))

            if thischunk_peak == 0:
                hnr_vector.append(0.5)
            else:
                chunk_len = len(chunk)
                hanning_window = np.hanning(chunk_len)
                chunk *= hanning_window
                # We start going ahead with FFT. Get n_fft.
                n_fft = 2 ** int(np.log2(chunk_len) + 1)

                hanning_window = np.hstack(
                    (hanning_window, np.zeros(n_fft - chunk_len))
                )
                chunk = np.hstack(
                    (chunk, np.zeros(n_fft - chunk_len))
                )
                ffts_outputs = []
                for fft_input in [chunk, hanning_window]:
                    fft_output = np.fft.fft(fft_input)
                    r = np.nan_to_num(
                        np.real(
                            np.fft.fft(fft_output * np.conjugate(fft_output))
                        )[: chunk_len]
                    )
                    ffts_outputs.append(r)
                r_x = ffts_outputs[0] / ffts_outputs[1]
                r_x /= r_x[0]

                indices = pu.indexes(r_x)

                # Now we index into r_x and into a linspace with these computed indices.
                time_array = np.linspace(0, thischunk_length, r_x.shape[0])

                myfilter = time_array[indices]
                candidate_values = r_x[indices]

                # Perform basic filtering according to period.
                # One side. 1.0 / max_pitch is min period.
                candidate_values = candidate_values[myfilter >= 1.0 / (sample_rate / 2.0)]
                # Update filter.
                myfilter = myfilter[myfilter >= 1.0 / (sample_rate / 2.0)]
                # Second side: 1.0 / min_pitch is max period.
                candidate_values = candidate_values[myfilter <= 1.0 / min_pitch]

                for i, v in enumerate(candidate_values):
                    if v > 1.0:
                        candidate_values[i] = 1.0 / v

                if candidate_values.shape[0] > 0:
                    strengths = [
                        np.max(candidate_values), np.max((
                            0, 2 - (thischunk_peak / waveform_peak) / silence_threshold
                        ))
                    ]
                    if np.argmax(strengths):
                        hnr_vector.append(0.5)
                    else:
                        hnr_vector.append(strengths[0])
                else:
                    hnr_vector.append(0.5)

    hnr_vector = np.array(hnr_vector)[np.array(hnr_vector) > 0.5]
    if hnr_vector.shape[0] == 0:
        return 0
    else:
        # Convert to dB.
        hnr_vector = 10.0 * np.log10(hnr_vector / (1.0 - hnr_vector))
    return np.mean(hnr_vector)
