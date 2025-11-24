import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class fbsde():
    def __init__(self, x_0, b, sigma, f, g, T, dim_x, dim_y, dim_d):
        self.x_0 = x_0.to(device)
        self.b = b
        self.sigma = sigma
        self.f = f
        self.g = g
        self.T = T
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_d = dim_d


class Model(nn.Module):
    def __init__(self, equation, dim_h):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(equation.dim_x + 1, dim_h)
        self.linear2 = nn.Linear(dim_h, dim_h)
        self.linear3 = nn.Linear(dim_h, equation.dim_y * equation.dim_d)
        self.y_0 = nn.Parameter(torch.rand(equation.dim_y, device=device))

        self.equation = equation

    def forward(self, batch_size, N, dW):
        def phi(x):
            x = F.relu(self.linear1(x))
            x = F.relu(self.linear2(x))
            return self.linear3(x).reshape(-1, self.equation.dim_y, self.equation.dim_d)  #[bs*(dy*dd)] -> [bs,dy,dd]

        delta_t = self.equation.T / N




        x = self.equation.x_0 + torch.zeros(batch_size, self.equation.dim_x, device=device)
        y = self.y_0 + torch.zeros(batch_size, self.equation.dim_y, device=device)

        x_seq = torch.empty(batch_size, N, self.equation.dim_x)
        y_seq = torch.empty(batch_size, N, self.equation.dim_y)
        z_seq = torch.empty(batch_size, N, self.equation.dim_y,self.equation.dim_d)




        for i in range(N-1):
            u = torch.cat((x, torch.ones(x.size()[0], 1, device=device) * delta_t * i), 1)
            z = phi(u)
            x_seq[:, i, :] = x
            y_seq[:, i, :] = y
            z_seq[:, i, :] = z
            w = dW[:, i, :].reshape(-1, self.equation.dim_d, 1) #reshape da iz [bs,dd]-> [bs,dd,1], we need additional 1 for matmul
            x = x + self.equation.b(delta_t * i, x, y) * delta_t + torch.matmul(self.equation.sigma(delta_t * i, x),
                                                                                w).reshape(-1, self.equation.dim_x)
            # mat. multiplication of sigma[bs,dx,dd][bs,dd,1] -> [bs,dx,1] reshape->[bs,dx]. In matmul computation is done batch wise
            y = y - self.equation.f(delta_t * i, x, y, z) * delta_t + torch.matmul(z, w).reshape(-1, self.equation.dim_y)

        x_seq[:, -1, :] = x
        y_seq[:,-1,:] = y
        return x, y, x_seq, y_seq, z_seq


class Solver():
    def __init__(self, equation, dim_h):
        self.model = Model(equation, dim_h).to(device)
        self.equation = equation

    #brez log
    def train(self, batch_size, N, itr, log=False):
        criterion = torch.nn.MSELoss().to(device)

        optimizer = torch.optim.Adam(self.model.parameters())

        loss_data, y0_data = [], [] ###leave this out

        for i in range(itr):
            dW = torch.randn(batch_size,N, self.equation.dim_d, device=device) * np.sqrt(self.equation.T/N)
            x, y, x_seq, y_seq, z_seq = self.model(batch_size, N,dW)
            loss = criterion(self.equation.g(x), y)
            loss_data.append(float(loss))
            y0_data.append(float(self.model.y_0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%100==0:
                print(i)

        if log:
            np.save('loss_data', loss_data)
            np.save('y0_data', y0_data)

        return loss_data, y0_data

##################################################
