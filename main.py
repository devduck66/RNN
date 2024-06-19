import pandas as pd


# Function to generate Fibonacci sequence
def fibonacci(n):
    fib_sequence = [0, 1]
    for i in range(2, n):
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return fib_sequence


# Number of columns and sequence length
num_columns = 50
sequence_length = 100

# Generate Fibonacci sequences
data = {f"col_{i}": fibonacci(sequence_length) for i in range(num_columns)}

# Create DataFrame and ensure numeric types
df = pd.DataFrame(data, dtype=float)
print(df.head())

import torch
from torch.utils.data import DataLoader, Dataset


# Custom dataset class
class FibonacciDataset(Dataset):
    def __init__(self, df, seq_length):
        self.data = df.values.astype(np.float32)  # Ensure data is float32
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length, :]
        y = self.data[idx + self.seq_length, :]
        return torch.tensor(x), torch.tensor(y)


# Sequence length
seq_length = 10

# Create dataset and dataloader
dataset = FibonacciDataset(df, seq_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 32).to(x.device)
        c0 = torch.zeros(2, x.size(0), 32).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Model parameters
input_size = num_columns
hidden_size = 32
num_layers = 2
output_size = num_columns

# Initialize model
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
