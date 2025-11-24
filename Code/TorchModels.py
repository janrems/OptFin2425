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


model = SimpleNet()
losses = train(model,epochs=1000)


x_eval, y_eval = generate_data(1000)
plot_results(model, x_eval, y_eval)
plot_losses(losses)


##################################################################
#time-dependent

class TimeDependentNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=1):
        super(TimeDependentNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, x, t):
        # Concatenate x and t along the last dimension
        inp = torch.cat([x, t], dim=1)
        out = self.activation(self.linear1(inp))
        out = self.activation(self.linear2(out))
        out = self.linear3(out)
        return out

def generate_data(n_samples=1000):
    x = 2 * torch.pi * torch.rand(n_samples, 1)  # x in [0, 2π]
    t = 2 * torch.pi * torch.rand(n_samples, 1)  # t in [0, 2π]
    y = torch.sin(x) * torch.cos(t)
    return x, t, y



def train(model, epochs=1000, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        x_train, t_train, y_train = generate_data()
        y_pred = model(x_train, t_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return losses


model = TimeDependentNet()
losses = train(model,epochs=1000)


plot_losses(losses)



######################################################
#Optimisation

def objective(x, y):
    return (x - torch.sin(y))**2

# Neural network u(y) ≈ optimal x
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, y ):
        return self.net(y)


def sample_y(n=1000):
    return (2 * torch.pi * torch.rand(n, 1)) - torch.pi

def train(model):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    losses = []

    for epoch in range(1000):
        y = sample_y(512)  # resample each step
        x = model(y)
        loss = objective(x, y).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss))

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss = {loss.item():.4f}")

    return losses

model = PolicyNet()
losses = train(model)

plot_losses(losses)

# Evaluate learned policy
y_test = torch.linspace(-torch.pi, torch.pi, 200).unsqueeze(1)
x_pred = model(y_test).detach()
x_true = torch.sin(y_test)

plt.plot(y_test.numpy(), x_true.numpy(), label="Target: sin(y)")
plt.plot(y_test.numpy(), x_pred.numpy(), label="NN output", linestyle="--")
plt.legend()
plt.grid(True)
plt.show()

