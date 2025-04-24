import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
X_train, Y_train = torch.load(
    "ctr_train.pt",
    map_location=device
)
X_val, Y_val = torch.load(
    "ctr_val.pt",
    map_location=device
)

print("the x train", X_train.size(0))
print("the y train", Y_train.size())

class CTRModel(nn.Module):
    def __init__(self):
        super(CTRModel, self).__init__()
        self.fc1 = nn.Linear(768, 128)     # Input size = 768 (384+384)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()        # Binary output

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)

model = CTRModel().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ Training loop
epochs = 10
batch_size = 8

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train.size(0))
    epoch_loss = 0.0

    for i in range(0, X_train.size(0), batch_size):
      indices = permutation[i:i + batch_size]
      batch_X, batch_Y = X_train[indices], Y_train[indices]

      outputs  = model(batch_X).squeeze()
      print("the float before", batch_Y)
      print("the after float", batch_Y.float())
      loss = criterion(outputs, batch_Y.float())

      # Backpropagation
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()
      print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), "ctr_trained_model.pt")
print("✅ Model saved to CTR_Model/ctr_trained_modal.pt")