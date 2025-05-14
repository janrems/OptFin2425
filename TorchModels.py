import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set manual seed for reproducibility
torch.manual_seed(42)

# Generate synthetic training data: (x, y) where y = sin(x) + noise
def generate_data(n_samples=500):
    x = torch.linspace(0, 2 * torch.pi, n_samples).unsqueeze(1)
    y = torch.sin(x) + 0.1 * torch.randn_like(x)
    return x, y

class SimpleNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        return x



def train(model, epochs=1000, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        x_train, y_train = generate_data()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return losses

def plot_results(model, x_train, y_train):
    model.eval()
    with torch.no_grad():
        y_pred = model(x_train)
    plt.figure(figsize=(8, 4))
    plt.plot(x_train.numpy(), y_train.numpy(), label="True", alpha=0.5)
    plt.plot(x_train.numpy(), y_pred.numpy(), label="Predicted", linewidth=2)
    plt.legend()
    plt.title("Function Approximation with Neural Network")
    plt.grid(True)
    plt.show()

def plot_losses(losses):
    plt.plot(losses)
    plt.show()

x_eval, y_eval = generate_data(1000)
model = SimpleNet()
losses = train(model,epochs=1000)

plot_results(model, x_eval, y_eval)
plot_losses(losses)

