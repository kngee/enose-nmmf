import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import serial

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
                    
                    print("Content detected: ", gas_class)

except KeyboardInterrupt:
    print("\nGas Detection Arborted")
finally:
    ser.close()



    