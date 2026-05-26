import os
import serial
import torch
import torch.nn as nn
import torch.nn.functional as F
from tkinter import filedialog, Tk
# 1. Open the file selection window for the model
root = Tk()
root.withdraw()  # Hide the blank background window
print("Please select your 'enose_ann_model.pt' file...")
model_path = filedialog.askopenfilename(
    title="Select the 'enose_ann_model.pt' Model File",
    filetypes=[("PyTorch Model Files", "*.pt"), ("All Files", "*.*")]
)
# Verify the user actually picked a file
if not model_path:
    print("No model file selected. Exiting script.")
    exit(1)
print(f"Loaded model file path: {model_path}")

# Set up the serial connection
ser = serial.Serial(
    port='COM9',       # Replace with your Arduino's serial port
    baudrate=9600,     # Match the baud rate in your Arduino code
)

# Create a Model Class that inherits nn.Module
class Model(nn.Module):

    # Construction of the neural network
    def __init__(self, in_features, h1=72, h2=72, h3=72, h4=72, h5=72, out_features=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)
        self.fc5 = nn.Linear(h4, h5)
        self.out = nn.Linear(h5, out_features)

    # Feed Forwarding Algorithm
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        x = F.tanh(self.fc5(x))
        x = self.out(x)
        return x


# Load checkpoint — contains weights, class names, and feature count
checkpoint  = torch.load(model_path, weights_only=False)
classes     = checkpoint['classes']       # e.g. ['acetone', 'air', 'hexanol']
n_features  = checkpoint['n_features']    # e.g. 8

print(f"Loaded model  →  {len(classes)} classes: {classes}")
print(f"                 {n_features} sensor features")

# Build model with the correct dimensions from the checkpoint
new_model = Model(in_features=n_features, out_features=len(classes))
new_model.load_state_dict(checkpoint['model_state_dict'])
new_model.eval()

# Read data boolean variable
read: bool = True

try:
    print("\nListening for Serial.write data...")
    while read:
        if ser.in_waiting > 0:

            data = ser.read(ser.in_waiting).decode('utf-8')  # Read and decode

            # Stop reading data
            if "Complete" in data:
                print("\nGas Detection Complete")
                read = False

            if "<WRITE>" in data:  # Only process Serial.write tagged data
                start = data.find("<WRITE>") + len("<WRITE>")
                end   = data.find("</WRITE>")
                filtered_data = data[start:end].strip()
                filtered_data = list(map(float, filtered_data.split(",")))

                # Print filtered data for debugging
                print(f"Detection Values: {filtered_data}")

                # Classify data based on the received inputs
                new_gas = torch.tensor(filtered_data, dtype=torch.float32)

                with torch.no_grad():
                    gas_eval  = new_model(new_gas).tolist()
                    gas_class = classes[gas_eval.index(max(gas_eval))]

                print("Content detected:", gas_class)

except KeyboardInterrupt:
    print("\nGas Detection Aborted")
finally:
    ser.close()