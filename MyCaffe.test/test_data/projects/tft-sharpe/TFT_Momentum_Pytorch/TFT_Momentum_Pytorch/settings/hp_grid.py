
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ray import tune

# Define your model using PyTorch
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 1)  # Assuming you have a regression task, change accordingly

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Define your hyperparameters
input_size =  # Define your input size
output_size = 1  # Define your output size
epochs = 10  # Define the number of training epochs

# Convert the data to PyTorch tensors and create DataLoader
# Assuming you have X_train, y_train, X_val, y_val as your training and validation data
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Use a fixed batch size for simplicity

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)  # Use a fixed batch size for simplicity

# Define a function for creating and training the model
def train_model(config, checkpoint_dir=None):
    model = MyModel(input_size, config['hidden_layer_size'], config['dropout_rate'])
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()  # Change this based on your task

    if checkpoint_dir:
        checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y.view(-1, 1))
            loss.backward()
            optimizer.step()

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            output = model(batch_X)
            val_loss += criterion(output, batch_y.view(-1, 1)).item()

    return val_loss

# Hyperparameter tuning with Ray Tune
search_space = {
    'hidden_layer_size': tune.grid_search([5, 10, 20, 40, 80, 160]),
    'dropout_rate': tune.grid_search([0.1, 0.2, 0.3, 0.4, 0.5]),
    'learning_rate': tune.loguniform(1e-4, 1e-1)
}

analysis = tune.run(
    train_model,
    config=search_space,
    num_samples=10,  # Number of hyperparameter combinations to try
    checkpoint_at_end=True,
    resources_per_trial={'cpu': 1},
)

best_config = analysis.get_best_config(metric="val_loss", mode="min")
print("Best Hyperparameters:", best_config)
