import re
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def plotATM(name):
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
        # Use regex to find numbers in the line
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

    for line in lines[5:]:  # Metadata starts from the 6th line
        # Skip empty or malformed lines
        if not line.strip():
            continue
        
        parts = line.strip().split("\t")
        if len(parts) < 5:  # Ensure there are at least 5 columns
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

    # Generate time vector
    time = np.arange(val.shape[1]) * sampling_interval

    # Plot signals
    plt.figure(figsize=(12, 8))
    for i in range(val.shape[0]):
        plt.plot(time, val[i, :], label=f"{signals[i]} ({units[i]})")
    plt.xlabel("Time (sec)")
    plt.ylabel("Amplitude")
    plt.title("ECG Signals")
    plt.legend()
    plt.grid(True)
    plt.show()

# Call the function with your data name (e.g., "101m")
plotATM("101_10sm")
