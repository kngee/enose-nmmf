import numpy as np
import serial
import time
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import csv
import os
import json

# Set up the serial connection
ser = serial.Serial(
    port='COM9',       # Replace with your Arduino's serial port
    baudrate=9600,     # Match the baud rate in your Arduino code
)

# Read data boolean variable
read : bool = True
# Lists to store each value for the array
MQ3 = []
MQ4 = []
MQ5 = []
MQ6 = [] 
MQ8 = [] 
MQ9 = []
MQ135 = [] 

# File name for the CSV
timestr = time.strftime("%Y%m%d-%H%M%S")
gas_name = input("What is the name of your gas?\n").capitalize()

folder_name = "Results/" + gas_name + "-" + timestr
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created.")
else:
    print(f"Folder '{folder_name}' already exists.")

base_dir = folder_name + "/"
csv_file = base_dir + gas_name + "-" + timestr + ".csv"
print("The file will be saved to: " + csv_file)

# We are required to read every row into a csv file from which we will be able to plot a normalized graph with respect to the minimum during the purge cycle which we consider to be the reference air voltage.
try:
    # Open the CSV file in write mode
    with open(csv_file, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["MQ135", "MQ3", "MQ6", "MQ9", "MQ5", "MQ8", "MQ4"])

        print("Listening for Serial.write data and saving to CSV")
        while read:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting).decode('utf-8')  # Read and decode
                
                if "<WRITE>" in data:  # Only process Serial.write tagged data
                    start = data.find("<WRITE>") + len("<WRITE>") 
                    end = data.find("</WRITE>")
                    filtered_data = data[start:end].strip()
                    filtered_data = list(map(float, filtered_data.split(",")))

                    # Store the data into respective sensor lists
                    MQ135.append(filtered_data[0])
                    MQ3.append(filtered_data[1])
                    MQ6.append(filtered_data[2])
                    MQ9.append(filtered_data[3])
                    MQ5.append(filtered_data[4])
                    MQ8.append(filtered_data[5])
                    MQ4.append(filtered_data[6])

                    # Write each value to a separate column in the CSV
                    csv_writer.writerow(filtered_data)

                    # Print for debugging
                    print(f"Detection Values: {filtered_data}")

except KeyboardInterrupt:
    print("\nGas Detection Aborted")
finally:
    ser.close()

# Combine MQ sensor data into a list for easier iteration
sensors = [
    (MQ135, "MQ135"),
    (MQ3, "MQ3"),
    (MQ6, "MQ6"),
    (MQ9, "MQ9"),
    (MQ5, "MQ5"),
    (MQ8, "MQ8"),
    (MQ4, "MQ4"),
]

# Create a single 1x1 plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Plot each sensor's data on the same axes
for sensor_data, label in sensors:
    min_val = min(sensor_data)
    max_val = max(sensor_data)
    norm = [(x - min_val) / (max_val - min_val) if max_val != min_val else 1.0 for x in sensor_data]
    ax.plot(norm, label=label)

# Labeling and legend
ax.set_title(gas_name + " combined MQ sensor response")
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Sensor Output Normalized to Air (V/V)")
ax.legend(loc='upper right')
ax.grid(True)

plt.tight_layout()
response_name = gas_name + "-resp-" + timestr + ".pdf"
plt.savefig(folder_name + "/" + response_name)  
plt.show()

# This smoothes the graph using Savitzky-Golay filter and plots the smoothed data on a new graph. 
# It also extracts the peaks and their properties for further analysis.
# The smoothed graph is saved as a PDF in the same folder as the raw response graph.
smooth_window = 10  # Adjust the window size as needed
smooth_poly = 4 # Adjust the polynomial order as needed

fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
# The smooth_norm dictionary stores the smoothed data for each sensor
smooth_norm = {}

for sensor_data, label in sensors:
    min_val = min(sensor_data)
    max_val = max(sensor_data)
    norm = np.array([(x - min_val) / (max_val - min_val) if max_val != min_val else 1.0 for x in sensor_data])
    if len(norm) >= smooth_window:
        smoothed = savgol_filter(norm, window_length=smooth_window, polyorder=smooth_poly)
    else:
        smoothed = norm  # If not enough data points, skip smoothing
    smooth_norm[label] = smoothed
    ax2.plot(smoothed, label=label)

ax2.set_title(gas_name + " smoothed combined MQ sensor response")
ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("Sensor Output Normalized to Air (V/V)")
ax2.legend(loc='upper right')
ax2.grid(True)

plt.tight_layout()
smooth_response_name = gas_name + "-smooth-resp-" + timestr + ".pdf"
plt.savefig(folder_name + "/" + smooth_response_name)
plt.show()


# Data extraction for peak detection
extraction_data = {"gas": gas_name, "timestamp": timestr, "sensors": {}}

txt_lines = [
    f"=== Data Extraction Report: {gas_name} ({timestr}) ===",
    "",
]

for label, smoothed_data in smooth_norm.items():
    peaks, props = find_peaks(smoothed_data, prominence=0.05)
    slope = np.diff(smoothed_data)

    sensor_report = {
        "peaks": [
            {"index": int(p), "value": round(float(smoothed_data[p]), 4),
             "prominence": round(float(props["prominences"][i]), 4)}
            for i, p in enumerate(peaks)
        ],
        "slope": {
            "max_rise_index": int(np.argmax(slope)),
            "max_rise_value": round(float(np.max(slope)), 4),
            "max_fall_index": int(np.argmin(slope)),
            "max_fall_value": round(float(np.min(slope)), 4),
            "mean": round(float(np.mean(slope)), 4),
            "rms": round(float(np.sqrt(np.mean(slope**2))), 4),
        }
    }
    extraction_data["sensors"][label] = sensor_report

    # Human-readable TXT block
    txt_lines += [
        f"[ {label} ]",
        f"  Peaks found : {len(peaks)}",
    ]
    for i, p in enumerate(peaks):
        txt_lines.append(
            f"    Peak {i+1}: index={p}, value={smoothed_data[p]:.4f}, "
            f"prominence={props['prominences'][i]:.4f}"
        )
    s = sensor_report["slope"]
    txt_lines += [
        f"  Max rise    : index={s['max_rise_index']}, slope={s['max_rise_value']}",
        f"  Max fall    : index={s['max_fall_index']}, slope={s['max_fall_value']}",
        f"  Mean slope  : {s['mean']}",
        f"  RMS slope   : {s['rms']}",
        "",
    ]

# Save structured JSON (machine-readable)
json_file = base_dir + f"{gas_name}-extraction-{timestr}.json"
with open(json_file, "w") as f:
    json.dump(extraction_data, f, indent=2)

# Save clean TXT (human-readable, one line per entry)
txt_file = base_dir + f"{gas_name}-extraction-{timestr}.txt"
with open(txt_file, "w") as f:
    f.write("\n".join(txt_lines))

print("\n".join(txt_lines))
print(f"\nJSON report saved → {json_file}")
print(f"TXT report saved  → {txt_file}")