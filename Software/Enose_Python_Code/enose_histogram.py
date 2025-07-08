''' CEFIM E-NOSE
 
  This file will generate a histogram showing the maximum normalized readings for a target gas from the commercial sensor array on the ENOSE
 
  v0.1
  2025/07/08
 
  Kenna Geleta
  u23575035@tuks.co.za

  Kerry Gilmore
  u23563525@tuks.co.za
 
  (C) 2025 University of Pretoria
 '''

''' CEFIM E-NOSE
 
  This file will generate a histogram showing the maximum normalized readings for a target gas from the commercial sensor array on the ENOSE
 
  v0.1
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

def histo_plot(csv_path):
    columns = [[] for _ in range(6)]
    labels = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  

        # Read header row and use first 6 columns as x-axis labels
        labels = header[:6]

        for row in reader:
            try:
                # Extract and convert first 6 columns to float
                values = list(map(float, row[:6]))
                for i in range(6):
                    columns[i].append(values[i])
            except ValueError:
                continue  # Skip rows with invalid data

    # Calculate max for each column
    columns = np.array(columns)
    print(columns)
    normalized = [col / (min(col)+1) for col in columns]
    max_values = [max(col) for col in normalized]

    # Plot histogram
    # Get title from file name (before first dash)
    base_name = os.path.basename(csv_path)
    title = base_name.split("-")[0]
    # Plot
    plt.figure(figsize=(8, 6))
    plt.bar(labels, max_values, color='skyblue', edgecolor='black')
    plt.title(f"MQ Sensor Values for {title}")
    plt.xlabel("Sensor")
    plt.ylabel("Maximum Value Normalized to Air (V/V)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# If you want to run this as a script:
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python histo_plot.py <path_to_csv>")
    else:
        histo_plot(sys.argv[1])