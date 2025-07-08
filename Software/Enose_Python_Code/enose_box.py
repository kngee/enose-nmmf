'''
 CEFIM E-NOSE

 This file will generate a box plot showing the normalized readings for a target gas
 from the commercial sensor array on the ENOSE.

 v0.3  
 2025/07/08

 Kenna Geleta  
 u23575035@tuks.co.za

 Kerry Gilmore  
 u23563525@tuks.co.za

 (C) 2025 University of Pretoria
'''

import matplotlib.pyplot as plt
import numpy as np
import csv
import sys
import os

def box_plot(csv_path):
    columns = [[] for _ in range(6)]
    labels = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        # Use the first 6 column headers as x-axis labels
        labels = header[:6]

        for row in reader:
            try:
                values = list(map(float, row[:6]))
                for i in range(6):
                    columns[i].append(values[i])
            except ValueError:
                continue  # Skip rows with bad data

    # Normalize each column to its own baseline (min + 1)
    normalized_columns = []
    for col in columns:
        min_val = min(col) if min(col) > 0 else 1
        normalized = [v / (min_val + 1) for v in col]
        normalized_columns.append(normalized)

    # Get plot title from filename (before the first dash)
    base_name = os.path.basename(csv_path)
    title = base_name.split("-")[0]

    # Plot box diagram
    plt.figure(figsize=(10, 6))
    plt.boxplot(normalized_columns, label=labels, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='black'),
                medianprops=dict(color='darkblue'),
                whiskerprops=dict(color='gray'),
                capprops=dict(color='gray'),
                flierprops=dict(marker='o', markerfacecolor='red', markersize=5, linestyle='none'))

    plt.title(f"MQ Sensor Box Plot for {title}")
    plt.xlabel("Sensor")
    plt.ylabel("Normalized Response to Air (V/V)")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Command-line interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python box_plot.py <path_to_csv>")
    else:
        box_plot(sys.argv[1])
