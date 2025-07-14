import numpy as np
import serial
import time
import matplotlib.pyplot as plt
import csv
import os

# Set up the serial connection
ser = serial.Serial(
    port='COM11',       # Replace with your Arduino's serial port
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