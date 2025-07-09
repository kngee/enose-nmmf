import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from   sklearn.model_selection import train_test_split

# Create a Model Class that inherits nn.Module
class Model(nn.Module):
    
    # Input layer (4 fetures of the flower) --> Hidden layers --> output (3 classes of the iris flower)
    
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

torch.manual_seed(30)

model = Model()

# Replace 'your_file.csv' with the path to your CSV file
df = pd.read_csv('enose.csv')

# Rename the names of the gases to values (Start labelling from 0).
# NOTE: When adding new gases e.g. butane, upadate both the csv file and the code
df['content'] = df['content'].replace('air', 0)
df['content'] = df['content'].replace('acetone', 1)
df['content'] = df['content'].replace('isopropyl', 2)


# Train Test Split
X = df.drop("content", axis=1)
Y = df['content']

# Convert these to numpy arrays
X = X.values
Y = Y.values

# Train test split (train with 20 % and test with 80 %)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=30)

# Convert X features to float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# Convert Y labels to tensors long
Y_train = torch.LongTensor(Y_train)
Y_test = torch.LongTensor(Y_test)

# Set the crietrion of the model to measure the error
criterion = nn.CrossEntropyLoss()
# Choose Adam Optimizer and set the learning rate 
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Train the model
epochs = 200        # (runs through all the training data in the network)
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

# Plot the epochs and the losses made
plt.plot(range(epochs), losses, label='Loss Curve')  # Add markers and a label for clarity
plt.ylabel("Loss/Epochs")
plt.xlabel("Epochs")
plt.title("Loss vs Epochs")  # Add a title for better understanding
plt.grid(True)  # Show the grid
plt.legend()  # Show the legend if you have multiple plots
plt.show()

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
new_gas = torch.tensor([146.0,420.0,29.0,213.0,262.0,250.0,41.0,130.0])
# Insert the data into the neural network 
with torch.no_grad():
    print(model(new_gas))


# Save the neural network model
torch.save(model.state_dict(), 'enose_ann_model.pt')

