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

