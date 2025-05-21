import matplotlib.pyplot as plt
import torch
from BSDESolver import fbsde
from BSDESolver import Model
from BSDESolver import Solver

dim_x, dim_y, dim_d,dim_j, dim_h, N, itr,lr, batch_size = 1, 1,1, 1,21, 50, 2000, 0.001, 2**8
x0_value, T = 1.0, 1.0
x_0 = torch.ones(dim_x)*x0_value




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

a=1
b_c=1
sig=1


def b(t, x, y):
    return a*(b_c - x)

def sigma(t, x):

    return sig*torch.sqrt(torch.abs(x)).reshape(-1, x.shape[1], dim_d)




def f(t, x, y, z):

    return -x*y

def g(x):

    return torch.ones(x.shape[0], x.shape[1])





equation = fbsde(x_0, b, sigma, f, g, T,dim_x, dim_y, dim_d)

bsde_solver = Solver(equation, dim_h)

loss, y0= bsde_solver.train(batch_size, N, 3)

plt.plot(loss)
plt.show()

true_y0 = 0.3965

plt.plot(y0)
plt.axhline(true_y0, color = "red")
plt.show()





