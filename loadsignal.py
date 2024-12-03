import re
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd

def plotECGFeatures(name):
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

    # Preprocess and clean data (if needed)
    val = np.where(val == -32768, np.nan, val)

    # Detect R-peaks (R-wave detection)
    r_peaks, _ = find_peaks(val[0, :], distance=sampling_frequency*0.6, height=0.5)  # Adjust parameters based on your data
    
    # Calculate RR intervals
    rr_intervals = np.diff(r_peaks) * sampling_interval  # Convert sample index to time
    
    # Detect QRS complexes and calculate QRS duration (approximation)
    qrs_onsets = r_peaks - int(sampling_frequency * 0.04)  # Approx. 40ms before R-wave
    qrs_offsets = r_peaks + int(sampling_frequency * 0.04)  # Approx. 40ms after R-wave
    
    qrs_duration = (qrs_offsets - qrs_onsets) * sampling_interval

    # Detect P and T waves (simplified detection based on R-peaks)
    p_wave_onsets = r_peaks - int(sampling_frequency * 0.12)  # Approx. 120ms before R-wave
    t_wave_offsets = r_peaks + int(sampling_frequency * 0.32)  # Approx. 320ms after R-wave
    
    # Calculate PR, QT, and P duration
    pr_interval = (r_peaks - p_wave_onsets) * sampling_interval
    qt_interval = (t_wave_offsets - r_peaks) * sampling_interval
    p_duration = (r_peaks - p_wave_onsets) * sampling_interval
    
    # Plot the ECG signal and extracted features
    time = np.arange(val.shape[1]) * sampling_interval
    
    # Plot ECG Signal
    plt.figure()
    plt.plot(time, val[0, :], label="ECG Signal")
    plt.title("ECG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show(block=False)

    # Plot R-peaks Detection
    plt.figure()
    plt.plot(r_peaks * sampling_interval, val[0, r_peaks], 'ro', label="R-peaks")
    plt.title("R-peaks Detection")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show(block=False)

    # Plot RR Intervals
    plt.figure()
    plt.plot(rr_intervals, label="RR Intervals (s)", color='g')
    plt.title("RR Intervals")
    plt.xlabel("R-peaks Index")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.show(block=False)

    # Plot QRS Duration
    plt.figure()
    plt.plot(qrs_duration, label="QRS Duration (s)", color='b')
    plt.title("QRS Duration")
    plt.xlabel("R-peaks Index")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.show(block=False)

    # Plot PR Interval
    plt.figure()
    plt.plot(pr_interval, label="PR Interval (s)", color='m')
    plt.title("PR Interval")
    plt.xlabel("R-peaks Index")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.show(block=False)

    # Plot QT Interval
    plt.figure()
    plt.plot(qt_interval, label="QT Interval (s)", color='c')
    plt.title("QT Interval")
    plt.xlabel("R-peaks Index")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.show(block=False)

    # Keep the windows open
    plt.show()

def saveECGFeaturesToExcel(name, output_filename="ECG_Features.xlsx"):
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

    # Preprocess and clean data (if needed)
    val = np.where(val == -32768, np.nan, val)

    # Detect R-peaks (R-wave detection)
    r_peaks, _ = find_peaks(val[0, :], distance=sampling_frequency*0.6, height=0.5)  # Adjust parameters based on your data
    
    # Calculate RR intervals
    rr_intervals = np.diff(r_peaks) * sampling_interval * 1000  # Convert sample index to milliseconds
    
    # Detect QRS complexes and calculate QRS duration (approximation)
    qrs_onsets = r_peaks - int(sampling_frequency * 0.04)  # Approx. 40ms before R-wave
    qrs_offsets = r_peaks + int(sampling_frequency * 0.04)  # Approx. 40ms after R-wave
    
    qrs_duration = (qrs_offsets - qrs_onsets) * sampling_interval * 1000  # Convert to milliseconds

    # Detect P and T waves (simplified detection based on R-peaks)
    p_wave_onsets = r_peaks - int(sampling_frequency * 0.12)  # Approx. 120ms before R-wave
    t_wave_offsets = r_peaks + int(sampling_frequency * 0.32)  # Approx. 320ms after R-wave
    
    # Calculate PR, QT, and P duration
    pr_interval = (r_peaks - p_wave_onsets) * sampling_interval * 1000  # Convert to milliseconds
    qt_interval = (t_wave_offsets - r_peaks) * sampling_interval * 1000  # Convert to milliseconds
    p_duration = (r_peaks - p_wave_onsets) * sampling_interval * 1000  # Convert to milliseconds
    
    # Ensure all arrays have the same length as r_peaks
    num_r_peaks = len(r_peaks)
    
    # Handle cases where features are shorter than the number of R-peaks
    rr_intervals = np.concatenate([np.nan * np.ones(1), rr_intervals])  # First RR interval is NaN
    qrs_duration = np.concatenate([np.nan * np.ones(1), qrs_duration])  # First QRS duration is NaN
    pr_interval = np.concatenate([np.nan * np.ones(1), pr_interval])  # First PR interval is NaN
    qt_interval = np.concatenate([np.nan * np.ones(1), qt_interval])  # First QT interval is NaN
    p_duration = np.concatenate([np.nan * np.ones(1), p_duration])  # First P duration is NaN
    qrs_onsets = np.concatenate([np.nan * np.ones(1), qrs_onsets])  # First QRS onset time is NaN
    qrs_offsets = np.concatenate([np.nan * np.ones(1), qrs_offsets])  # First QRS offset time is NaN
    p_wave_onsets = np.concatenate([np.nan * np.ones(1), p_wave_onsets])  # First P wave onset time is NaN
    t_wave_offsets = np.concatenate([np.nan * np.ones(1), t_wave_offsets])  # First T wave offset time is NaN

    # Adjust the lengths of the arrays to match the length of r_peaks
    max_len = num_r_peaks
    rr_intervals = rr_intervals[:max_len]
    qrs_duration = qrs_duration[:max_len]
    pr_interval = pr_interval[:max_len]
    qt_interval = qt_interval[:max_len]
    p_duration = p_duration[:max_len]
    qrs_onsets = qrs_onsets[:max_len]
    qrs_offsets = qrs_offsets[:max_len]
    p_wave_onsets = p_wave_onsets[:max_len]
    t_wave_offsets = t_wave_offsets[:max_len]

    # Prepare data for Excel
    data = {
        "R_PEAK_INDEX": r_peaks * sampling_interval * 1000,  # Convert to milliseconds
        "R_PEAK_TIME (ms)": r_peaks * sampling_interval * 1000,  # Convert to milliseconds
        "RR_INTERVAL (ms)": rr_intervals,
        "QRS_ONSET_TIME (ms)": qrs_onsets * sampling_interval * 1000,  # Convert to milliseconds
        "QRS_OFFSET_TIME (ms)": qrs_offsets * sampling_interval * 1000,  # Convert to milliseconds
        "QRS_DURATION (ms)": qrs_duration,
        "PR_INTERVAL (ms)": pr_interval,
        "QT_INTERVAL (ms)": qt_interval,
        "P_DURATION (ms)": p_duration,
        "P_WAVE_ONSET_TIME (ms)": p_wave_onsets * sampling_interval * 1000,  # Convert to milliseconds
        "T_WAVE_OFFSET_TIME (ms)": t_wave_offsets * sampling_interval * 1000  # Convert to milliseconds
    }

    # Convert to pandas DataFrame
    df = pd.DataFrame(data)

    # Export DataFrame to Excel
    df.to_excel(output_filename, index=False)
    print(f"ECG feature data has been saved to {output_filename}")

# Call the function with your data name (e.g., "101m") and specify output Excel file name
saveECGFeaturesToExcel("101m", "ECG_Features_101m_ms.xlsx")
# Call the function with your data name (e.g., "101m")
plotECGFeatures("101m")
