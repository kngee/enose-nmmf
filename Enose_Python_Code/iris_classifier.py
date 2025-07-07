import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Create a Model Class that inherits nn.Module
class Model(nn.Module):
    
    # Contruction of the neural network
    def __init__(self, in_features=4, h1=8, h2=8, h3=8, out_features=3):
        super().__init__()                      # instance our nn.Module
        self.fc1 = nn.Linear(in_features, h1)      
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, out_features)

    # Feed Forwarding Algorithm
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.out(x)

        return x

# Load the saved model
new_model = Model()
new_model.load_state_dict(torch.load('iris_model_ann.pt', weights_only=True))

# Ensure that the netwrok is correctly loaded
print(new_model.eval())

# Classify data based on the received inputs
new_iris = torch.tensor([7.0, 3.2, 4.7, 1.4])
# Insert the data into the neural network 
with torch.no_grad():
    print(new_model(new_iris))