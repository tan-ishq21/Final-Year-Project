import re
import numpy as np
import scipy.io
import pywt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def denoise_ecg(signal, wavelet='db4', levels=10):
    """
    Denoises the ECG signal using wavelet transform.
    Removes low-frequency baseline drift and high-frequency noise.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=levels)
    # Remove high-frequency noise and baseline drift
    coeffs[0] = np.zeros_like(coeffs[0])  # Baseline drift
    coeffs[-1] = np.zeros_like(coeffs[-1])  # High-frequency noise
    return pywt.waverec(coeffs, wavelet)


def segment_ecg(signal, r_peaks, fs, pre_r=0.25, post_r=0.38):
    """
    Segments ECG signal into individual beats based on R-peaks.
    Extracts a window of `pre_r` seconds before and `post_r` seconds after each R-peak.
    Returns a list of segments.
    """
    segments = []
    for r in r_peaks:
        start = max(0, int(r - pre_r * fs))
        end = min(len(signal), int(r + post_r * fs))
        segments.append(signal[start:end])
    return segments  # Return as a list


def plotATM(name, pre_r=0.25, post_r=0.38):
    # File paths
    mat_name = f"{name}.mat"
    info_name = f"{name}.info"

    # Load the .mat file
    mat_data = scipy.io.loadmat(mat_name)
    
    # 'val' is expected to be the key for the signal matrix in the .mat file
    if 'val' not in mat_data:
        print(f"Key 'val' not found in {mat_name}. Available keys: {mat_data.keys()}")
        return
    val = mat_data['val']  # Matrix containing signal values

    # Read .info file
    with open(info_name, 'r') as f:
        lines = f.readlines()

    # Extract sampling interval and frequency from the fourth line using regex
    sampling_line = lines[3].strip()
    try:
        sampling_frequency = float(re.search(r"Sampling frequency: (\d+)", sampling_line).group(1))
        sampling_interval = float(re.search(r"Sampling interval: ([\d.]+)", sampling_line).group(1))
    except (AttributeError, ValueError):
        print(f"Error parsing the sampling line: {sampling_line}")
        return

    # Parse signal metadata
    signals = []
    gains = []
    baselines = []
    units = []

    for line in lines[5:]:
        if not line.strip():
            continue
        parts = line.strip().split("\t")
        if len(parts) < 5:
            print(f"Skipping malformed line: {line.strip()}")
            continue
        signals.append(parts[1])
        gains.append(float(parts[2]))
        baselines.append(float(parts[3]))
        units.append(parts[4])

    gains = np.array(gains)
    baselines = np.array(baselines)

    # Replace missing values (-32768) with NaN
    val = np.where(val == -32768, np.nan, val)

    # Baseline correction and scaling
    for i in range(val.shape[0]):
        val[i, :] = (val[i, :] - baselines[i]) / gains[i]

    # Apply denoising to each signal
    for i in range(val.shape[0]):
        val[i, :] = denoise_ecg(val[i, :])

    # Segment the signal based on R-peaks
    for i in range(val.shape[0]):
        # Detect R-peaks
        r_peaks, _ = find_peaks(val[i, :], distance=sampling_frequency * 0.6)  # Minimum distance ~0.6s
        segments = segment_ecg(val[i, :], r_peaks, fs=sampling_frequency, pre_r=pre_r, post_r=post_r)

        # Plot the original signal with R-peaks in a separate window
        plt.figure()  # New window for each plot
        time = np.arange(val[i, :].shape[0]) / sampling_frequency
        plt.plot(time, val[i, :], label=f"ECG Signal ({signals[i]})")
        plt.plot(r_peaks / sampling_frequency, val[i, r_peaks], "ro", label="R-peaks")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.title(f"ECG Signal with R-peaks - {signals[i]} ({units[i]})")
        plt.legend()
        plt.grid(True)

        # Plot segmented beats in another separate window
        plt.figure()  # Another new window for segmented beats
        for j, beat in enumerate(segments[:10]):  # Plot up to 10 segments
            time_axis = np.linspace(-pre_r, post_r, len(beat))  # Adjust time axis per segment length
            plt.plot(time_axis, beat + j * 0.5, label=f"Segment {j+1}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.title(f"Segmented Beats - {signals[i]}")
        plt.legend()
        plt.grid(True)

    # Display all the plots at once
    plt.show()


# Call the function with your data name (e.g., "101m")
plotATM("101_10sm")
