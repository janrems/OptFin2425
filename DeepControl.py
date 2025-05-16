import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

#############################################
#optimal portfolio
T = 1
N = 50
dt = T/N
t=torch.linspace(0,T,N)
sqrdt = np.sqrt(dt)
x_0 = 1

drift = 0.05
volatility = 0.2



def b(x,t, u):
    return drift*x*u

def sigma(x,t, u):
    return volatility*x*u


class log_utility(nn.Module):
    def __init__(self, dim_h, batch_size, N):
        super(log_utility, self).__init__()
        self.dim_h = dim_h
        self.batch_size = batch_size
        self.N = N
        self.linear1 = nn.Linear(2, dim_h)
        self.linear2 = nn.Linear(dim_h, dim_h)
        self.linear3 = nn.Linear(dim_h, 1)
        self.activation = nn.ReLU()


    def forward(self, dW):
        def phi(x):
            x = self.activation(self.linear1(x))
            x = self.activation(self.linear2(x))
            return self.linear3(x)

        output_seq = torch.empty((self.batch_size,self.N,1))

        input_seq = torch.empty((self.batch_size,self.N,1))


        x = torch.ones(self.batch_size,1)*x_0



        for i in range(N):
            input_seq[:,i,:] = x
            inpt = torch.cat((x, torch.ones_like(x) * dt * i), dim=-1)
            u = phi(inpt)
            output_seq[:, i,:] = u
            if i < N-1:
                x = x + b(x,i*dt,u) * dt + sigma(x,i*dt,u)*dW[:,i+1,:]


        return x, input_seq, output_seq



def loss_ln(terminal):
    return - torch.mean(torch.log(torch.linalg.vector_norm(terminal, dim=-1)))



def train(itr, model):
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    losses = []

    for n in range(itr):
        dW = torch.randn((model.batch_size,model.N,1))*np.sqrt(dt)
        xT, state_seq, control_seq = model(dW)

        loss = loss_ln(xT)
        losses.append(float(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if n%100 == 0:
            print(n)

    return losses, control_seq, state_seq


itr = 1000
batch_size = 5000
dim_h=32
#itr = 2000
#batch_size = 5000
#dim_h = 32
model = log_utility(dim_h, batch_size, N)
losses, control_p, state_seq_p = train(itr,model)


opt = drift/volatility**2 *np.ones(N)

i = np.random.randint(1000)
plt.plot(t,control_p[i,:,0].detach().numpy(), label="Predicted control")
plt.plot(t,opt, label="True control")
plt.plot(t,state_seq_p[i,:,0].detach().numpy(), label="State")
plt.legend()
plt.show()

plt.plot(losses)
plt.show()



#################################
#consumption
T = 1
N = 50
dt = T/N
t=torch.linspace(0,T,N)
x_0 = 1
K = 1



def b(x,t,u):
    return (0.05-u)*x

def sigma(x,t,u):
    return 0.2*x

class consumption(nn.Module):
    def __init__(self, dim_h, batch_size, N):
        super(consumption, self).__init__()
        self.dim_h = dim_h
        self.batch_size = batch_size
        self.N = N
        self.linear1 = nn.Linear(2, dim_h)
        self.linear2 = nn.Linear(dim_h, dim_h)
        self.linear3 = nn.Linear(dim_h, 1)
        self.softplus = nn.Softplus(beta=1,threshold=3)
        self.activation = nn.ReLU()



    def forward(self, dW):
        def phi(x):
            x = self.activation(self.linear1(x))
            x = self.activation(self.linear2(x))
            x = self.softplus(self.linear3(x))
            return x


        output_seq = torch.empty((self.batch_size,
                                  self.N,1))

        input_seq = torch.empty((self.batch_size,
                                  self.N,1))



        x = torch.ones(self.batch_size,1)*x_0

        int = torch.zeros(self.batch_size,1)

        for i in range(N):
            input_seq[:,i, :] = x
            u = torch.cat((x, torch.ones_like(x) * dt * i), 1)
            out = phi(u)
            output_seq[:,i, :] = out
            if i < N-1:
                x = x + b(x,i*dt,out) * dt + sigma(x,i*dt,out)*dW[:,i+1,:]
                if torch.sum(x<=0)>0:
                    print(str(itr) + "x <=0")
                int = int + torch.log(out*x)*dt


        input_seq[:,-1,:]=x
        return x, input_seq, output_seq, int



def loss_con(terminal, int):
    return - torch.mean(int[:,0] + K*torch.log(terminal[:,0]))



def train(itr, model):
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses = []

    for n in range(itr):
        dW = torch.randn(model.batch_size, model.N, 1)*np.sqrt(dt)
        xT, state_seq, control_seq, int = model(dW)

        loss = loss_con(xT, int)
        losses.append(float(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if n%100 == 0:
            print(n)

    return losses, control_seq, state_seq


itr = 1000
batch_size = 3000
dim_h=32
#itr = 2000
#batch_size = 5000
#dim_h = 32
model = consumption(dim_h, batch_size, N)
losses, control_p, state_seq_p = train(itr,model)


opt = 1/(2-t)
i = np.random.randint(1000)
plt.plot(t,control_p[i,:,0].detach().numpy(), label="Predicted control")
plt.plot(t,opt, label="True control")
plt.plot(t,state_seq_p[i,:,0].detach().numpy(), label="State")
plt.legend()
plt.show()

plt.plot(losses)
plt.show()
