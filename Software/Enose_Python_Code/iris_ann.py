import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


# Create a Model Class that inherits nn.Module
class Model(nn.Module):
    
    # Input layer (4 fetures of the flower) --> Hidden layers --> output (3 classes of the iris flower)
    
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

torch.manual_seed(32)

model = Model()

# Replace 'your_file.csv' with the path to your CSV file
df = pd.read_csv('iris.csv')

# Rename the names of the plants to values (Start labelling from 0).
df['species'] = df['species'].replace('setosa', 0)
df['species'] = df['species'].replace('versicolor', 1)
df['species'] = df['species'].replace('virginica', 2)


# Train Test Split
X = df.drop("species", axis=1)
Y = df['species']

# Convert these to numpy arrays
X = X.values
Y = Y.values

# Train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=32)

# Convert X features to float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# Convert Y labels to tensors long
Y_train = torch.LongTensor(Y_train)
Y_test = torch.LongTensor(Y_test)

# Set the crietrion of the model to measure the error
criterion = nn.CrossEntropyLoss()
# Choose Adam Optimizer and set the learning rate 
optimizer = torch.optim.Adam(model.parameters(), lr=0.015)

# Train the model
epochs = 100        # (runs through all the training data in the network)
losses = []         # A list of the losses made through the training
for i in range(epochs):
    # Go forward and get a predicition
    Y_pred = model.forward(X_train)       # Get the predicted results
    # Measure the loss/error (Initally high)
    loss = criterion(Y_pred, Y_train)
    # Keep track of the losses
    losses.append(loss.detach().numpy())

    if i%10 == 0:
        print(f'Epoch: {i} and loss: {loss}')

    # Implement back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plot
# plt.plot(range(epochs), losses, marker='o', label='Loss Curve')  # Add markers and a label for clarity
# plt.ylabel("Loss/Epochs")
# plt.xlabel("Epochs")
# plt.title("Loss vs Epochs")  # Add a title for better understanding
# plt.grid(True)  # Show the grid
# plt.legend()  # Show the legend if you have multiple plots
# plt.show()

# Evalulate model on the test data (basically turn off back propagation)
with torch.no_grad():
    Y_eval = model.forward(X_test)
    loss = criterion(Y_eval, Y_test)

print(f"Losses: {loss}")

correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        Y_val = model.forward(data)

        # Tells us what type of flower class our netwrok think it is
        print(f'{i+1} : {str(Y_val)} \t Predicted: {Y_test[i]} \t Expected: {Y_val.argmax().item()}')

        # Evaluate if the predicted class is correct or not
        if Y_val.argmax().item() == Y_test[i]:
            correct += 1

print(f'Correct predictions: {correct}')



# Classify data based on the received inputs
new_iris = torch.tensor([4.7, 3.2, 1.3, 0.2])
# Insert the data into the neural network 
with torch.no_grad():
    print(model(new_iris))


# Save the neural network model
torch.save(model.state_dict(), 'iris_model_ann.pt')

# Load the saved model
new_model = Model()
new_model.load_state_dict(torch.load('iris_model_ann.pt', weights_only=True))

# Ensure that the netwrok is correctly loaded

print(new_model.eval())
