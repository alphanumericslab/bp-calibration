import os
import wfdb
import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate, find_peaks, filtfilt

data_dir = '../../../../../../data/physionet.org/files/mimicdb/1.0.0'


def reject_segment(segment, Pmax, Pmin=25, NumConstantSamples=50):
    """
    Evaluates a segment of blood pressure (BP) values to determine if it should be rejected
    based on specific criteria related to constant values, maximum and minimum pressure thresholds,
    and the systolic blood pressure (SBP) value.

    Parameters:
    segment (np.ndarray): Array of BP values over time.
    Pmax (float): Maximum pressure threshold. Segments with pressure values higher than Pmax will be rejected.
    Pmin (float, optional): Minimum pressure threshold. Default is 25.
    NumConstantSamples (int, optional): Number of constant samples to consider a segment as constant. Default is 50.

    Returns:
    bool: True if the segment should be rejected, False otherwise.
    """
    # Check if the samples of the segment are constant over a period of time.
    isConstant = np.any(
        np.convolve(np.double(np.diff(segment) == 0), np.ones(NumConstantSamples), 'valid') == NumConstantSamples)
    if isConstant:
        # print('Constant Samples')
        return True

    # Cuff pressure should be higher than SBP to block the flow and also
    # lower than DBP to capture DBP at the end.
    pressureCondition = Pmin <= np.min(segment) and Pmax >= np.max(segment)
    if not pressureCondition:
        # print('Risk of Saturation')
        return True

    # If the SBP value is too low, it is likely that the segment is not valid.
    if np.max(segment) <= 70:
        # print('Very Low SBP')
        return True

    # If all the conditions above were satisfied, keep the segment.
    return False


def find_intersect(y1, y2):
    """
    Finds the intersection points of two curves represented by arrays y1 and y2.

    Parameters:
    y1 (np.ndarray): An array of doubles representing the first curve.
    y2 (np.ndarray): An array of doubles representing the second curve.

    Returns:
    np.ndarray: An array of indices where the intersection points occur between y1 and y2.
    """
    return np.where(np.diff(np.sign(y1 - y2)) != 0)[0]


def extract_segment(subject, initPressure, segLen=60, fs=125):
    """
    Extracts valid segments of arterial blood pressure (ABP) signals from a given subject's data.

    Parameters:
    subject (int): Subject number. In mimicdb, subject recording directory names are subject numbers.
    initPressure (float): Initial pressure used as Pmax while validating the segment.
    segLen (int, optional): Length of the segment in seconds. Default is 60.
    fs (int, optional): Sampling frequency of the signal. Default is 125.

    Returns:
    list: A list of valid segments for the given subject. Each segment is a numpy array.
    """
    segments = []
    N = int(np.ceil(segLen * fs))

    # read the record
    subject_dir = str(subject) + '/' + str(subject)
    record_dir = os.path.join(data_dir, subject_dir)
    signals = wfdb.rdsamp(record_dir)

    # extract ABP signal channel from other modalities
    abp_ch = signals[1]['sig_name'].index('ABP')
    abp_signal = signals[0][:, abp_ch]

    # number of segments in subject recordings
    nSeg = int(np.floor(len(abp_signal) / N))

    for k in range(nSeg):
        segment = abp_signal[k * N:(k + 1) * N]
        if not reject_segment(segment, initPressure):
            segments.append(segment)

    return segments


def estimate_SDBP(segments, cuffPressure, decompose_error=False, fs=125, extrema_window=0.5):
    """
    Estimates systolic and diastolic blood pressure (SBP and DBP) from given segments of arterial blood pressure (ABP)
    signals and corresponding cuff pressure values.

    Parameters:
    segments (list of np.ndarray): List of segments, each represented as a numpy array.
    cuffPressure (np.ndarray): Array of cuff pressure values based on time.

    Returns:
    BP (np.ndarray): A numpy array containing the SBP and DBP values along with their estimated values.
    """
    SBP = []
    DBP = []
    SBP_hat = []
    DBP_hat = []
    err_cuff_SBP = []
    err_cuff_DBP = []
    err_resp_SBP = []
    err_resp_DBP = []

    wLen = int(extrema_window * fs)

    for segment in segments:
        # find the intersection of BP with external cuff pressure
        indices = find_intersect(segment, cuffPressure)

        # skip segment if there is not any intersection
        if len(indices) == 0:
            continue

        # save true and estimated values for SDBP
        SBP.append(np.max(segment))
        DBP.append(np.min(segment))
        SBP_hat.append(cuffPressure[indices[0]])
        DBP_hat.append(cuffPressure[indices[-1]])

        if decompose_error:
            # Detect local systolic peaks and diastolic troughs
            SBP_local = np.max(segment[indices[0]:indices[0]+wLen])
            DBP_local = np.min(segment[indices[-1]-wLen:indices[-1]])

            # Calculate the error decomposition
            err_cuff_SBP.append(SBP_local - SBP_hat[-1])
            err_cuff_DBP.append(DBP_local - DBP_hat[-1])
            err_resp_SBP.append(SBP[-1] - SBP_local)
            err_resp_DBP.append(DBP[-1] - DBP_local)


    BP = np.vstack((SBP, DBP, SBP_hat, DBP_hat))
    if not decompose_error:
        return BP
    else:
        err = np.vstack((err_cuff_SBP, err_cuff_DBP, err_resp_SBP, err_resp_DBP))
        return BP, err



def env_detector(sig, winLen=10, fs=125):
    """Extract adaptive beat envelopes from one contiguous ABP segment.

    Local percentile amplitude sets prominence; autocorrelation estimates beat
    spacing within shared search bounds of 30--180 beats/minute. Troughs are
    minima between systolic peaks. PCHIP avoids cubic interpolation overshoot.
    Unsupported edges and rejected beat intervals are NaN. These heuristics
    are not a validated artifact or arrhythmia classifier.
    """
    sig = np.asarray(sig, dtype=float)
    if sig.ndim != 1:
        raise ValueError("sig must be one-dimensional")
    if not np.isfinite(fs) or fs <= 0 or not np.isfinite(winLen) or winLen <= 0:
        raise ValueError("fs and winLen must be positive and finite")
    env_u = np.full(sig.shape, np.nan)
    env_d = np.full(sig.shape, np.nan)
    if len(sig) < max(4, int(4 * fs)) or not np.all(np.isfinite(sig)):
        return env_u, env_d
    if np.ptp(sig) <= np.finfo(float).eps:
        return env_u, env_d

    # Smooth only for detection; preserve original pressures at extrema.
    smooth = uniform_filter1d(sig, size=max(1, int(round(0.04 * fs))),
                              mode='nearest')
    window = min(len(sig), max(3, int(round(winLen * fs))))
    centered = smooth - uniform_filter1d(smooth, size=window, mode='nearest')
    centers = np.arange(0, len(sig), max(1, window // 2))
    amplitude = []
    for center in centers:
        local = centered[max(0, center - window // 2):
                         min(len(sig), center + window // 2 + 1)]
        lo, hi = np.percentile(local, [10, 90])
        amplitude.append(hi - lo)
    scale = np.interp(np.arange(len(sig)), centers, amplitude)
    if np.max(scale) <= np.finfo(float).eps:
        return env_u, env_d

    min_lag = max(1, int(np.ceil(fs / 3)))
    max_lag = min(len(sig) - 1, int(2 * fs))
    ac = correlate(centered, centered, mode='full', method='fft')[len(sig)-1:]
    ac = ac / np.arange(len(sig), 0, -1)
    candidates, _ = find_peaks(ac)
    candidates = candidates[(candidates >= min_lag) & (candidates <= max_lag)]
    if not len(candidates):
        return env_u, env_d
    # Prefer the earliest strong periodicity over a multiple of the beat period.
    strongest = np.max(ac[candidates])
    if strongest <= 0:
        return env_u, env_d
    period = candidates[ac[candidates] >= 0.9 * strongest][0]
    peaks, _ = find_peaks(
        smooth, prominence=np.maximum(0.3 * scale, np.finfo(float).eps),
        distance=max(1, int(0.55 * period)))
    if len(peaks) < 3:
        return env_u, env_d

    radius = max(1, int(round(0.04 * fs)))
    peaks = np.unique([
        max(0, p - radius) + np.argmax(sig[max(0, p-radius):p+radius+1])
        for p in peaks
    ])
    intervals = np.diff(peaks)
    good = ((intervals >= min_lag) & (intervals <= max_lag) &
            (intervals >= 0.5 * period) & (intervals <= 1.8 * period))
    troughs = np.array([a + np.argmin(sig[a:b+1])
                       for a, b in zip(peaks[:-1], peaks[1:])])
    pulse = np.minimum(sig[peaks[:-1]], sig[peaks[1:]]) - sig[troughs]
    good &= (troughs > peaks[:-1]) & (troughs < peaks[1:])
    good &= pulse >= 0.3 * scale[troughs]

    # Never interpolate across rejected beats or extrapolate unsupported edges.
    edges = np.diff(np.r_[False, good, False].astype(int))
    for first, stop in zip(np.flatnonzero(edges == 1),
                           np.flatnonzero(edges == -1)):
        upper = peaks[first:stop+1]
        lower = troughs[first:stop]
        if len(lower) < 2:
            continue
        samples = np.arange(lower[0], lower[-1] + 1)
        env_u[samples] = PchipInterpolator(upper, sig[upper])(samples)
        env_d[samples] = PchipInterpolator(lower, sig[lower])(samples)
    return env_u, env_d


def pwelch(x, window, noverlap, nfft, do_filter=True):
    """
    Estimate power spectral density using Welch's method.

    This function implements Welch's method for estimating the power spectral density
    of a signal. It divides the signal into overlapping segments, applies a window
    function, and computes the average periodogram.

    Parameters:
    x (array-like): Input signal.
    window (array-like): Window function to apply to each segment.
    noverlap (int): Number of samples to overlap between segments.
    nfft (int): Number of FFT points.

    Returns:
    array: Power spectral density estimate.

    The function performs the following steps:
    1. Calculates the number of segments based on the signal length, window length, and overlap.
    2. Checks for and excludes constant segments (std < 1e-6).
    3. For each valid segment:
       - Extracts the segment
       - Applies the window function
       - Normalizes the windowed segment
       - Computes the FFT
       - Updates the power spectral density estimate
    4. Returns the final power spectral density estimate.
    """

    N = len(x)
    W = len(window)
    nSeg = int(np.floor((N - W) / (W - noverlap)) + 1)

    # check if the segment is constant
    K = nSeg
    for i in range(nSeg):
        # extract i-th segment
        segment = x[i * (W - noverlap):i * (W - noverlap) + W]
        if np.std(segment) < 1e-6:
            K -= 1

    PSD = np.zeros(nfft // 2)
    for i in range(nSeg):
        # extract i-th segment
        segment = x[i * (W - noverlap):i * (W - noverlap) + W]

        # skip constant segments
        if np.std(segment) < 1e-6:
            continue

        # apply windowing
        segment_windowed = segment * window

        # Filter out freq contents between 0-0.25 Hz
        if do_filter:
            segment_filtered = segment_windowed - lp_filter_zero_phase(segment_windowed, 0.05 / 125)
        else:
            segment_filtered = segment_windowed

        # normalize segments
        segment_norm = (segment_filtered - np.mean(segment_filtered)) / np.std(segment_filtered)

        # calculate FFT
        segment_fft = np.fft.fft(segment_norm, n=nfft)

        # update PSD
        PSD = PSD + (1 / K) * (np.abs(segment_fft[:nfft // 2]) ** 2 / (np.sum(window ** 2)))

    return PSD


def bias_correction_LSE(x):
    """
    Perform bias correction using Least Squares Error (LSE) estimation.

    Parameters:
    x (array-like): Input data, 2D array.

    Returns:
    array: For single observation, returns input unchanged.
           For multiple observations, returns mean across observations.

    Note: Assumes input contains two variables (e.g., SBP and DBP) across observations.
    """
    x = x.reshape(2, -1)  # reshape to avoid dimension conflicts
    xbar = np.mean(x, axis=1).reshape(2, -1)
    return xbar


def bias_correction_ML(x, mu_w):
    """
    Perform bias correction using Maximum Likelihood (ML) approach.

    Parameters:
    x (array-like): Input data, 2D array.
    mu_w (array-like): Mean measurement noise.

    Returns:
    array: Bias-corrected data. For single observation, subtracts mu_w from input.
           For multiple observations, subtracts mu_w from mean across observations.

    Note: Assumes input contains two variables (e.g., SBP and DBP) across observations.
    """
    x = x.reshape(2, -1)  # reshape to avoid dimension conflicts
    mu_w = mu_w.reshape(2, -1)
    xbar = np.mean(x, axis=1).reshape(2, -1)
    return xbar - mu_w


def bias_correction_Bayes(x, mu_w, C_w, mu_theta, C_theta):
    """
    Perform bias correction using Maximum Likelihood (ML) approach.

    Parameters:
    x (array-like): Input data, 2D array.
    mu_w (array-like): Mean measurement noise.

    Returns:
    array: Bias-corrected data. For single observation, subtracts mu_w from input.
           For multiple observations, subtracts mu_w from mean across observations.

    Note: Assumes input contains two variables (e.g., SBP and DBP) across observations.
    """
    x = x.reshape(2, -1)  # reshape to avoid dimension conflicts
    N = x.shape[1]
    mu_theta = mu_theta.reshape(2, -1)
    mu_w = mu_w.reshape(2, -1)
    xbar = np.mean(x, axis=1).reshape(2, -1)
    return mu_theta + C_theta @ np.linalg.inv(C_theta + C_w/N) @ (xbar - (mu_w + mu_theta))


def png_2_eps(fig_path):
    """
    Convert a PNG image to EPS format.

    Parameters:
    fig_path (str): Path to the input PNG file.

    Functionality:
    - Handles transparent backgrounds by filling with white.
    - Converts image to RGB mode.
    - Saves as EPS with 300 DPI and 95% quality.

    Output:
    - Saves EPS file in the same directory as input, with the same filename (*.eps).
    - Prints success message upon completion.

    Note: Resizing functionality is commented out in the current version.
    """

    png_image = Image.open(fig_path)
    if png_image.mode in ('RGBA', 'LA') or (png_image.mode == 'P' and 'transparency' in png_image.info):
        alpha = png_image.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", png_image.size, (255, 255, 255) + (255,))
        bg.paste(png_image, mask=alpha)
        png_image = bg.convert('RGB')
    else:
        png_image = png_image.convert('RGB')

    # new_size = (int(png_image.width * 0.7), int(png_image.height * 0.7))
    # png_image = png_image.resize(new_size, Image.LANCZOS)
    eps_path = os.path.join(os.path.dirname(fig_path), os.path.basename(fig_path)[:-4] + '.eps')
    png_image.save(eps_path, format='EPS', dpi=(300, 300), quality=95)
    print('EPS file is saved successfully!')


def lp_filter_zero_phase(x, fc):
    """
    lp_filter_zero_phase - Second-order zero-phase Lowpass filter.

    Syntax: y = lp_filter_zero_phase(x, fc)

    Inputs:
      x: Vector or matrix of input data (channels x samples).
      fc: -3dB cut-off frequency normalized by the sampling frequency.

    Output:
      y: Vector or matrix of filtered data (channels x samples).

      Revision History:
          2006: First release
          2023: Renamed from deprecated version LPFilter()

      Read more:  Mitra, S. (2010). Digital signal processing (4th ed.)
                  New York, NY: McGraw-Hill Professional.

    Revision History:
        2023: Translated to Python from Matlab

    Amulya Jain, 2023
    The Open-Source Electrophysiological Toolbox
    https://github.com/alphanumericslab/OSET
    """

    if fc > 1:
        raise ValueError("fc must be less than 1")
    x = np.array([x], dtype=np.double)
    k = 0.7071  # Cut-off value of 1/sqrt(2) or -6dB amplitude attenuation
    alpha = (
        1
        - k * np.cos(2 * np.pi * fc)
        - np.sqrt(
            (
                2 * k * (1 - np.cos(2 * np.pi * fc))
                - k**2 * np.sin(2 * np.pi * fc) ** 2
            )
        )
    ) / (1 - k)
    y = np.zeros_like(x)
    for i in range(x.shape[0]):
        y[i, :] = filtfilt([1 - alpha], [1, -alpha], x[i, :], padlen=3)
    return y[0]
