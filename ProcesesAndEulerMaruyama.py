import torch
import numpy as np
import matplotlib.pyplot as plt

##############################################################
#1.1

T = 1.0
N = 500
dt =T/N
t = torch.linspace(0,T,N)

n_samples = 10

W1 = torch.zeros(n_samples,N)


for i in range(N-1):
    nor_t = torch.randn(n_samples)*np.sqrt(dt)
    W1[:,i+1] = W1[:,i] + nor_t


dW = torch.randn((n_samples,N))*np.sqrt(dt)
W = torch.cumsum(dW,dim=1)
W = torch.roll(W,1)
W[:,0] = torch.zeros(n_samples)

i = np.random.randint(n_samples)
plt.plot(t,W1[i,:])
plt.plot(t,W[i,:])
plt.show()

############################################################
#1.2
n_samples = 1000

dW = torch.randn((n_samples,N))*np.sqrt(dt)
dW[:,0] = torch.zeros(n_samples)
W = torch.cumsum(dW,1)

time = t.unsqueeze(0)
time = time.repeat(n_samples,1)


M = torch.exp(W-0.5*time)

EM = torch.mean(M,dim=0)

i_to_plot = np.random.randint(0,n_samples,20)

plt.plot(t,EM, color="red")
for i in i_to_plot:
    plt.plot(t,M[i,:],color="blue",alpha=0.11)
plt.show()

########################################################
#1.3

n_samples = 1000
T = 1.0
N = 10000
dt =T/N
t = torch.linspace(0,T,N)

def gen_bm(n_samples,N,T):
    dt=T/N
    dW = torch.randn((n_samples, N)) * np.sqrt(dt)
    W = torch.cumsum(dW, dim=1)
    W = torch.roll(W, 1)
    W[:, 0] = torch.zeros(n_samples)

    return W

W = gen_bm(n_samples,N,T)

M = torch.zeros(n_samples,N)

for sample in range(n_samples):
    max = 0.0
    for tstep in range(N):
        current = W[sample,tstep]
        if  current > max:
            M[sample,tstep] = current
            max = current
        else:
            M[sample,tstep] = max

i = np.random.randint(n_samples)
plt.plot(t,W[i,:])
plt.plot(t,M[i,:])
plt.show()


Wnp = W.numpy()

Mnp = np.maximum.accumulate(Wnp,axis=1)

i = np.random.randint(n_samples)
plt.plot(t,Mnp[i,:])
plt.plot(t,Wnp[i,:])
plt.show()

from scipy import stats

Wabs = np.abs(Wnp)

plt.plot(t,Wabs[i,:])
plt.show()



stats.ks_2samp(Wabs[:,400], Mnp[:,400])[1]

reject = []
for n in range(N):
    p = stats.ks_2samp(Wabs[:,n], Mnp[:,n])[1]
    if p < 0.05:
        reject.append(1)
    else:
        reject.append(0)


plt.plot(t,np.array(reject))
plt.show()

for j in range(30):
    plt.plot(Mnp[j,:],color="blue",alpha=0.81)
    plt.plot(Wabs[j, :], color="red", alpha=0.81)
plt.show()


n = 7000
plt.hist(Mnp[:,n],color="blue")
plt.hist(Wabs[:,n], color="red")
plt.show()

###################################
#2.1


n_samples = 10000
T = 1.0
N = 500
dt =T/N
t = torch.linspace(0,T,N)
rate = 10.0

#first approach that uses the fact that Poisson process has stationary independent increments

def gen_poisson1(n_samples,T,N, rate):
    dt = T/N
    rates = torch.ones(n_samples,N)*(rate*dt)
    poiss_increments = torch.poisson(rates)
    hpp = torch.cumsum(poiss_increments,dim=1)
    hpp = torch.roll(hpp, 1, dims=1)
    hpp[:,0] = torch.zeros(n_samples)
    return hpp




hpp = gen_poisson1(n_samples,T,N,rate)

i = np.random.randint(n_samples)
plt.plot(t,hpp[i,:])
plt.show()

####2. approach: jump times are uniformly distributed



def gen_poisson2(n_samples,T,N,rate):
    p = torch.distributions.Poisson(torch.tensor([rate*T]))
    hpp = torch.zeros(n_samples,N)
    u = torch.distributions.Uniform(torch.zeros(1),torch.ones(1))

    for s in range(n_samples):
        m = p.sample()
        times = u.sample((int(m),))
        for time in times:
            indx = int(time / dt)
            hpp[s,indx] = 1

    return torch.cumsum(hpp,dim=1)


hpp2 = gen_poisson2(n_samples,T,N, rate)

i = np.random.randint(n_samples)
plt.plot(t,hpp2[i,:])
plt.show()


reject = []
for n in range(N):
    p = stats.ks_2samp(hpp[:,n], hpp2[:,n])[1]
    if p < 0.05:
        reject.append(1)
    else:
        reject.append(0)


plt.plot(t,np.array(reject))
plt.show()

### 3. approach: time between jumps is exponentialy distributed

def gen_poisson3(n_s,T,N,rate):
    e = torch.distributions.Exponential(torch.tensor([rate]))
    dt = T/N
    hpp = torch.zeros(n_s, N)
    for s in range(n_s):
        clock = e.sample()
        while clock < T:
            indx = int(clock / dt)
            hpp[s,indx] = 1
            clock += e.sample()

    return torch.cumsum(hpp, dim=1)


hpp3 = gen_poisson3(n_samples,T,N, rate)

i = np.random.randint(n_samples)
plt.plot(t,hpp3[i,:])
plt.show()



#######################################################
#3.1

n_samples = 1000
T = 10.0
N = 10000
dt =T/N
t = torch.linspace(0,T,N)

x0 = 0
theta = 2
mu = 0.5
sigma = 0.2


def gen_ou(n_s,T,N,W):
    dt = T/N
    x = torch.ones(n_s, N)*x0
    for i in range(N-1):
        dx = theta*(mu-x[:,i])*dt + (sigma)*(W[:,i+1]-W[:,i])
        x[:,i+1] = x[:,i] + dx
    return x

W = gen_bm(n_samples,N,T)
x = gen_ou(n_samples,T,N,W)

Ex = torch.mean(x,dim=0)

i = np.random.randint(n_samples)
plt.plot(t,x[i,:])
plt.plot(t,Ex)
plt.show()




x_5 = x[:,int(5/dt)]

prob = 0
for s in range(n_samples):
    if x_5[s] <= mu:
        prob = prob + 1

prob = prob/n_samples


time = t.unsqueeze(0)
time =time.repeat(n_samples,1)


def gen_int(n_s,T,N,W):
    dt = T/N
    x = torch.ones(n_s, N)*0
    for i in range(N-1):
        s = i*dt
        dx = np.exp(theta*s)*(W[:,i+1]-W[:,i])
        x[:,i+1] = x[:,i] + dx
    return x

int = gen_int(n_samples,T,N,W)
x_true = x0*torch.exp(-theta*time) + mu*(1-torch.exp(-theta*time)) + sigma*torch.exp(-theta*time)*int

i = np.random.randint(n_samples)
plt.plot(t,x[i,:],color='blue',label="EM")
plt.plot(t,x_true[i,:], color="red",label="true")
plt.legend(loc='best')
plt.show()



###########################################
#3.2
def plot2(x,y):
    i = np.random.randint(n_samples)
    plt.plot(t,x[i,:], color="red")
    plt.plot(t,y[i,:], color="blue")
    plt.show()

n_samples = 100000
N = 2**9
T = 1.0
dt = T/N
t_orig = torch.linspace(0,T,N)



W = gen_bm(n_samples,N,T)

x0 = 1
mu = 0.3
sigma=0.2

def gen_explicit(t,W):
    t_tensor = t.unsqueeze(0)
    t_tensor.shape
    t_tensor = t_tensor.repeat(n_samples, 1)
    t_tensor.shape
    x = x0 * torch.exp((mu - 0.5 * sigma ** 2) * t_tensor + sigma * W)
    return x

errors = {"dt":[], "str1":[], "str2":[], "weak":[]}

for j in range(10):
    t = t_orig[::j+1]
    dt = float(t[1]-t[0])
    Wtmp = W[:,::j+1]
    print("len t="+str(t.shape) + "  W len="+ str(Wtmp.shape))
    x_exp = gen_explicit(t, Wtmp)
    x = torch.ones(n_samples,len(t))*x0
    for i in range(len(t)-1):
        dx = x[:,i]*(mu*dt + sigma*(Wtmp[:,i+1]- Wtmp[:,i]))
        x[:,i+1] = x[:,i] + dx
    plot2(x_exp,x)
    abs = torch.abs(x_exp-x)

    err1 = torch.mean(abs,dim=0)
    err1 = torch.max(err1)
    errors["str1"].append(float(err1))

    err2 = torch.max(abs,dim=1).values
    err2 = torch.mean(err2)
    errors["str2"].append(float(err2))

    err3 = torch.abs(torch.mean(x_exp[:,-1]) - torch.mean(x[:,-1]))
    errors["weak"].append(float(err3))

    errors["dt"].append(dt)


c1 = np.sqrt(errors["dt"])[0]/np.array(errors["str1"])[0]
c2 = np.sqrt(errors["dt"])[0]/np.array(errors["str2"])[0]
c3 = np.array(errors["dt"])[0]/np.array(errors["weak"])[0]


plt.plot(errors["dt"],c1*np.array(errors["str1"]))
plt.plot(errors["dt"], np.sqrt(errors["dt"]),'--')
plt.show()

plt.plot(errors["dt"],c2*np.array(errors["str2"]))
plt.plot(errors["dt"], np.sqrt(errors["dt"]),'--')
plt.show()

plt.plot(errors["dt"],18*np.array(errors["weak"]))
plt.plot(errors["dt"], errors["dt"],'--')
plt.show()

plt.plot(errors["dt"],errors["weak"],'r')
plt.plot(errors["dt"],errors["str1"],'g')
plt.plot(errors["dt"],errors["str2"],'b')
plt.show()













