"""
 CEFIM E-NOSE

 This file will generate a grouped histogram showing the maximum normalized readings
 for all target gases from the commercial sensor array on the ENOSE.

 v0.2
 2025/07/08

 Kenna Geleta
 u23575035@tuks.co.za

 Kerry Gilmore
 u23563525@tuks.co.za

 (C) 2025 University of Pretoria
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def load_max_normalized_values(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        labels = header[:6]
        columns = [[] for _ in range(6)]

        for row in reader:
            try:
                values = list(map(float, row[:6]))
                for i in range(6):
                    columns[i].append(values[i])
            except ValueError:
                continue

        columns = np.array(columns)
        # Normalize each sensor column to air (min + 1)
        normalized = [col / (min(col) + 1) for col in columns]
        max_values = [max(col) for col in normalized]

    return labels, max_values

def grouped_bar_histogram(training_folder):
    all_files = [f for f in os.listdir(training_folder) if f.endswith('.csv')]
    all_gas_names = []
    all_max_values = []
    labels = []

    for file in sorted(all_files):
        gas_name = file.split("-")[0]
        path = os.path.join(training_folder, file)

        labels, max_vals = load_max_normalized_values(path)
        all_gas_names.append(gas_name)
        all_max_values.append(max_vals)

    all_max_values = np.array(all_max_values)  # shape: (num_gases, num_sensors)
    num_gases, num_sensors = all_max_values.shape
    bar_width = 0.12

    x = np.arange(num_sensors)  # base x-locations for sensor groups

    plt.figure(figsize=(10, 6))

    for i in range(num_gases):
        plt.bar(x + i * bar_width,
                all_max_values[i],
                width=bar_width,
                label=all_gas_names[i],
                edgecolor='black')

    plt.xlabel("Sensor")
    plt.ylabel("Max Normalized Reading (V/V)")
    plt.title("Max Normalized Sensor Readings per Gas")
    plt.xticks(x + (bar_width * (num_gases - 1) / 2), labels)
    plt.legend(title="Gas Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Run the plot function
if __name__ == "__main__":
    grouped_bar_histogram("Training")
