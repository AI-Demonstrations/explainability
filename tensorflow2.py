import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import math

# 1. Define a simple Neural Network
class SineApproximator(nn.Module):
    def __init__(self):
        super(SineApproximator, self).__init__()
        # A simple network: 1 input (x), two hidden layers, 1 output (y)
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    # 2. Initialize Model, Loss Function, and Optimizer
    model = SineApproximator()
    criterion = nn.MSELoss() # Mean Squared Error measures how far off our predictions are
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 3. Initialize TensorBoard Writer
    writer = SummaryWriter('runs/neural_network_sine')

    # Let's log the actual model architecture to TensorBoard!
    # We pass a dummy input through the model so TensorBoard can trace the operations.
    dummy_input = torch.zeros(1, 1)
    writer.add_graph(model, dummy_input)

    print("Training neural network and logging to TensorBoard...")

    # 4. The Training Loop
    epochs = 1000
    for epoch in range(epochs):
        # Generate a batch of 32 random x values between -PI and PI
        x = (torch.rand(32, 1) * 2 * math.pi) - math.pi
        
        # The true y values we want the network to learn
        y_true = torch.sin(x)

        # Forward pass: ask the model to predict y based on x
        y_pred = model(x)
        
        # Calculate the error (loss)
        loss = criterion(y_pred, y_true)

        # Backward pass: update the neural network weights to improve accuracy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 5. Log progress to TensorBoard every 10 epochs
        if epoch % 10 == 0:
            # Log the decreasing error
            writer.add_scalar('Training/Loss', loss.item(), epoch)
            
            # Log the model's current guess for a specific value: x = pi/2
            # The exact sine of pi/2 is 1.0, so we want to watch this prediction get closer to 1.0
            test_x = torch.tensor([[math.pi / 2]])
            test_pred = model(test_x).item()
            writer.add_scalar('Model_Predictions/Sine_of_Pi_over_2', test_pred, epoch)

    # Always close the writer!
    writer.close()
    print("Training complete! You can now start the TensorBoard server.")

if __name__ == "__main__":
    main()