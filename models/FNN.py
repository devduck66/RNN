import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Example data (features and labels)
features = torch.randn(1000, 10)  # 1000 samples, 10 features each
labels = torch.randn(1000, 1)  # 1000 labels

# Normalize the data
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split data into training and validation sets
features_train, features_val, labels_train, labels_val = train_test_split(
    features, labels, test_size=0.2
)

# Convert numpy arrays to torch tensors
features_train = torch.tensor(features_train, dtype=torch.float32)
labels_train = torch.tensor(labels_train, dtype=torch.float32)
features_val = torch.tensor(features_val, dtype=torch.float32)
labels_val = torch.tensor(labels_val, dtype=torch.float32)

# Create DataLoader
batch_size = 32
train_dataset = TensorDataset(features_train, labels_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(features_val, labels_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


# Initialize model, criterion, and optimizer
model = SimpleNN(input_size=10, hidden_size=64, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", patience=10, verbose=True
)


# Early stopping
class EarlyStopping:
    def __init__(self, patience=20, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# Training loop
num_epochs = 1000
early_stopping = EarlyStopping(patience=20)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_idx, (features, labels) in enumerate(train_loader):
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(features)  # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the weights
        epoch_loss += loss.item()

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for features, labels in val_loader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(
        f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}"
    )

    scheduler.step(val_loss)

    if early_stopping(val_loss):
        print("Early stopping")
        break

print("Training completed.")
