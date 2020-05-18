import numpy as np


def get_deviation_for_dfa(signal, window_length):
    """Given a signal, compute the trend value for one window length, as per
    https://link.springer.com/article/10.1186/1475-925X-6-23
    In order to get the overall DFA (detrended fluctuation analysis),
    compute this for a variety of window lengths, then plot that on a
    log-log graph, and get the slope.

    Args:
        signal (np.array, [T, ]): waveform
        window_length (int > 0): L in the paper linked above. Length of windows for trend.

    Returns:
        float: average rmse for fitting lines on chunks of window lengths on the
            cumulative sums of this signal.
    """
    rmse = 0
    # Step 1, integrate time series (cumulative sum)
    Y = np.cumsum(signal)
    # Step 2, separate into chunks of length window_length +/- 1.
    chunks = np.array_split(Y, np.ceil(Y.shape[0] / window_length))
    # Step 3, fit a line for all of these chunks.
    for chunk in chunks:
        slope, offset = np.polyfit(np.arange(chunk.shape[0]), chunk, 1)
        rmse += np.sqrt(
            (((offset + slope * np.arange(chunk.shape[0])) - chunk) ** 2).sum()
        )
    rmse /= len(chunks)
    return rmse


def get_dfa(signal, window_lengths):
    """Given a signal, compute the DFA (detrended fluctuation analysis)
    as per https://link.springer.com/article/10.1186/1475-925X-6-23
    See paper equations (13) to (16) for more information.
    """
    fl_list = []
    for length in window_lengths:
        fl_list.append(get_deviation_for_dfa(signal, length))

    log_l = [np.log(length) for length in window_lengths]
    log_fl = [np.log(fl) for fl in fl_list]

    # Fit a line.
    slope, _ = np.polyfit(log_l, log_fl, 1)
    return slope
