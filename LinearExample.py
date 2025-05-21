import matplotlib.pyplot as plt
import torch
import numpy as np
from BSDESolver import fbsde
from BSDESolver import Model
from BSDESolver import Solver

dim_x, dim_y, dim_d,dim_j, dim_h, N, itr,lr, batch_size = 1, 1,1, 1,21, 50, 2000, 0.001, 2**8
x0_value, T = 1.0, 1.0
x_0 = torch.ones(dim_x)*x0_value




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

theta = 0.2



def b(t, x, y):
    return torch.zeros_like(x)

def sigma(t, x):
    return (theta*x).reshape(-1, x.shape[1], dim_d)




def f(t, x, y, z):
    return theta*z.reshape(-1,y.shape[1])

def g(x):
    return x





equation = fbsde(x_0, b, sigma, f, g, T,dim_x, dim_y, dim_d)

bsde_solver = Solver(equation, dim_h)

loss, y0= bsde_solver.train(batch_size, N, itr)

plt.plot(loss)
plt.show()


################################################

dW = torch.randn(batch_size, N, equation.dim_d, device=device) * np.sqrt(T/N)

bsde_solver.model.eval()
_, _, x_seq, y_seq, z_seq = bsde_solver.model(batch_size,N,dW)

#Works for 1-D case
W = torch.cumsum(dW, dim=1)
W = torch.roll(W,1,1)
W[:,0,:] = torch.zeros(batch_size,equation.dim_d)

t = torch.linspace(0,T,N)
time = torch.broadcast_to(t,(batch_size,equation.dim_d,N))
time = time.transpose(1,-1)




true_y_seq = torch.exp(theta*W + (T-1.5*time)*theta**2)

true_y0 = true_y_seq[0,0,0]

plt.plot(y0)
plt.axhline(true_y0, color = "red")
plt.show()

i =np.random.randint(batch_size)
plt.plot(t,y_seq[i,:,0].detach().numpy(), label="Estimated")
plt.plot(t,true_y_seq[i,:,0], label="True")
plt.legend()
plt.show()




