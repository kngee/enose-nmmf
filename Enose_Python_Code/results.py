import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import serial
import matplotlib.pyplot as plt

# Set up the serial connection
ser = serial.Serial(
    port='COM5',       # Replace with your Arduino's serial port
    baudrate=9600,     # Match the baud rate in your Arduino code
)

# Create a Model Class that inherits nn.Module
class Model(nn.Module):
    
    # Contruction of the neural network
    def __init__(self, in_features=8, h1=72, h2=72, h3=72, h4=72, h5=72, out_features=3):
        super().__init__()                      # instance our nn.Module
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

# Read data boolean variable
read : bool = True
# Lists to store each value for the array
MQ2 = [] 
MQ3 = []
MQ4 = []
MQ5 = []
MQ6 = [] 
MQ8 = [] 
MQ9 = []
MQ135 = [] 
G = []

# Load the saved model
new_model = Model()
new_model.load_state_dict(torch.load('enose_ann_model.pt', weights_only=True))

try:

    print("Listening for Serial.write data...")
    while read:
        if ser.in_waiting > 0:
            
            data = ser.read(ser.in_waiting).decode('utf-8')  # Read and decode
            
            # Stop reading data
            if data == "Complete":
                print("\nGas Detection Complete")
                read = False
        
            if "<WRITE>" in data:  # Only process Serial.write tagged data
                start = data.find("<WRITE>") + len("<WRITE>")
                end = data.find("</WRITE>")
                filtered_data = data[start:end].strip()
                filtered_data = list(map(float, filtered_data.split(",")))

                # Store the data from the 
                MQ135.append(filtered_data[0])
                MQ3.append(filtered_data[1])
                MQ2.append(filtered_data[2])
                MQ6.append(filtered_data[3])
                MQ9.append(filtered_data[4])
                MQ5.append(filtered_data[5])
                MQ8.append(filtered_data[6])
                MQ4.append(filtered_data[7])

                # Print filtered data for debugging
                print(f"Detection Values: {filtered_data}")

                # Classify data based on the received inputs
                new_gas = torch.tensor(filtered_data)
                # Insert the data into the neural network 
                with torch.no_grad():
                    
                    # Convert the classifcation tensor to a list
                    gas_eval = new_model(new_gas).tolist()          
                    # Identify the gas detected
                    gas_class = ""
                    if gas_eval.index(max(gas_eval)) == 0:
                        gas_class = 'air'
                    elif gas_eval.index(max(gas_eval)) == 1:
                        gas_class = 'acetone'
                    elif gas_eval.index(max(gas_eval)) == 2:
                        gas_class = 'isopropyl'
                    # Store the gas index
                    G.append(gas_eval.index(max(gas_eval))*50)
                    print("Content detected: ", gas_class)

except KeyboardInterrupt:
    print("\nGas Detection Arborted")
finally:
    ser.close()


# # Example data for MQ sensor values and G (replace with actual data)
# MQ2 = np.random.rand(10)  # Replace with actual MQ2 data
# MQ3 = np.random.rand(10)  # Replace with actual MQ3 data
# MQ4 = np.random.rand(10)  # Replace with actual MQ4 data
# MQ5 = np.random.rand(10)  # Replace with actual MQ5 data
# MQ6 = np.random.rand(10)  # Replace with actual MQ6 data
# MQ8 = np.random.rand(10)  # Replace with actual MQ8 data
# MQ9 = np.random.rand(10)  # Replace with actual MQ9 data
# MQ135 = np.random.rand(10)  # Replace with actual MQ135 data
# G = np.random.randint(0, 3, 10)  # Replace with actual G data (e.g., classifications)

# Combine MQ sensor data into a list for easier iteration
sensors = [
    (MQ135, "MQ135"),
    (MQ3, "MQ3"),
    (MQ2, "MQ2"),
    (MQ6, "MQ6"),
    (MQ9, "MQ9"),
    (MQ5, "MQ5"),
    (MQ8, "MQ8"),
    (MQ4, "MQ4"),
]

# Create the 4x2 grid of subplots
fig, axes = plt.subplots(4, 2, figsize=(8, 8))

# Flatten the 2D axes array for easy iteration
axes = axes.flatten()

# Plot each sensor data and G on its own subplot
for i, (sensor_data, title) in enumerate(sensors):
    axes[i].plot(sensor_data, label=f"{title} Sensor Data", color=f"C{i}")  # Sensor data
    axes[i].plot(G, label="Gas Detection (G)", linestyle='--', color="black")  # G data
    axes[i].set_title(f"{title} vs G")
    axes[i].grid(True)
    axes[i].legend(loc="upper left")

    # Only add y-axis labels for the left column
    if i % 2 == 0:
        axes[i].set_ylabel("Sensor/G Value")

    # Only add x-axis labels for the bottom row
    if i >= 6:
        axes[i].set_xlabel("Time Step")

# Adjust layout to avoid overlapping
plt.tight_layout()

# Display the plot
plt.show()