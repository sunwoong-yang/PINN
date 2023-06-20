
from ..pde.Burger import *

#@title Problem Def
def pde1(x, y):
  mu = 0.01 / torch.pi
  y_x = torch.autograd.grad(y.sum(), x, create_graph=True)[0][:,[0]]
  y_x2 = torch.autograd.grad(y_x.sum(), x, create_graph=True)[0][:,[0]] # 2nd derivative (https://discuss.pytorch.org/t/second-order-derivatives-of-loss-function/71797/3)
  y_t = torch.autograd.grad(y.sum(), x, create_graph=True)[0][:,[1]]
  # pde = y_t - y_x2 - torch.exp(-x[:,[1]]) * (-torch.sin(torch.pi*x[:,[0]]) + torch.pi**2 * torch.sin(torch.pi*x[:,[0]]))
  pde = y_t + y * y_x - mu * y_x2
  return pde

BCIC_func1 = lambda x : np.zeros((x.shape[0],1))
BCIC_func2 = lambda x : np.zeros((x.shape[0],1))
BCIC_func3 = lambda x : -np.sin(np.pi * x[:,0]).reshape(-1,1)


pde = [pde1]
bcic = [(BCIC_func1,"D"), (BCIC_func2,"D"), (BCIC_func3,"D")]
pde_n, bc_n =1, 2

#data 정의 : [pde_collocation, bc1, bc2,..., ic1, ic2,...]
domain_bound = [[-1,1], [0,1]]
pts_collo = 1000
pts_bcic = 200
data_collocation = sampling(domain_bound, pts_collo)
data_bcic1 = np.concatenate([domain_bound[0][0] * np.ones((pts_bcic,1)), np.random.uniform(low=domain_bound[1][0], high=domain_bound[1][1], size=(pts_bcic,1))], axis=1)
data_bcic2 = np.concatenate([domain_bound[0][1] * np.ones((pts_bcic,1)), np.random.uniform(low=domain_bound[1][0], high=domain_bound[1][1], size=(pts_bcic,1))], axis=1)
data_bcic3 = np.concatenate([np.random.uniform(low=domain_bound[0][0], high=domain_bound[0][1], size=(pts_bcic,1)), np.zeros((pts_bcic,1))], axis=1)
data = [data_collocation, data_bcic1, data_bcic2, data_bcic3]

# net = Net(inputs=2, outputs=1, hidden_layers=[64]*4, activation='tanh')
# net = net.to(device)
# mse_cost_function = torch.nn.MSELoss() # Mean squared error
# optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
# PINN_dwx = PINN(net, data, pde, bcic, bc_n)
# PINN_dwx.compile(optimizer)
# epochs = 15000
# for epoch in range(epochs):
#   PINN_dwx.train_step(his=1000, dynamic = False)

# PINN_dwx.train_step(his=True, dynamic=False, L_BFGS=15000)

net = Net(inputs=2, outputs=1, hidden_layers=[25]*4, activation='tanh')
net = net.to(device)
mse_cost_function = torch.nn.MSELoss() # Mean squared error
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
PINN_dwo = PINN(net, data, pde, bcic, bc_n)
PINN_dwo.compile(optimizer)
epochs = 15000
for epoch in range(epochs):
  PINN_dwo.train_step(his=1000, dynamic = ["v1",1])

# PINN_dwo.train_step(his=True, dynamic=["v1",1], L_BFGS=15000)

#@title Finite Difference
import scipy
from scipy.integrate import odeint
# https://github.com/sachabinder/Burgers_equation_simulation/blob/main/Burgers_solver_SP.py
############## SET-UP THE PROBLEM ###############

mu = 1
nu = 0.01 / np.pi #kinematic viscosity coefficient

#Spatial mesh
x_max, x_min = 1, -1  #Range of the domain according to x [m]
dx = 0.001 #Infinitesimal distance
N_x = int((x_max-x_min)/dx) #Points number of the spatial mesh
X = np.linspace(x_min,x_max,N_x) #Spatial array

#Temporal mesh
L_t = 1 #Duration of simulation [s]
dt = 0.001  #Infinitesimal time
N_t = int(L_t/dt) #Points number of the temporal mesh
T = np.linspace(0,L_t,N_t) #Temporal array

#Wave number discretization
k = 2*np.pi*np.fft.fftfreq(N_x, d = dx)


#Def of the initial condition
u0 = -np.sin(np.pi * X) #Single space variable fonction that represent the wave form at t = 0
# viz_tools.plot_a_frame_1D(X,u0,0,L_x,0,1.2,'Initial condition')

############## EQUATION SOLVING ###############

#Definition of ODE system (PDE ---(FFT)---> ODE system)
def burg_system(u,t,k,mu,nu):
    #Spatial derivative in the Fourier domain
    u_hat = np.fft.fft(u)
    u_hat_x = 1j*k*u_hat
    u_hat_xx = -k**2*u_hat

    #Switching in the spatial domain
    u_x = np.fft.ifft(u_hat_x)
    u_xx = np.fft.ifft(u_hat_xx)

    #ODE resolution
    u_t = -mu*u*u_x + nu*u_xx
    return u_t.real


#PDE resolution (ODE system resolution)
U = odeint(burg_system, u0, T, args=(k,mu,nu,), mxstep=5000).T

test_resi = 200

plt.figure(figsize = (20, 4))
for t__ in [0,0.2,0.4,0.6,0.8,1]:
  x_test=np.linspace(-1,1,test_resi)
  t_test = np.ones_like(x_test) * t__
  x_test, t_test = torch.tensor(x_test).to(device), torch.tensor(t_test).to(device)
  x_test_ = torch.concat([(x_test.reshape(-1,1)).float(),(t_test.reshape(-1,1)).float()],dim=1)
  u_ = PINN_dwx.predict(x_test_).cpu().detach().numpy()
  plt.plot(x_test.cpu(),u_,'r')
  if t__ != 1:
    plt.plot(X, U[:,int(t__*N_t)],'--k')
  else:
    plt.plot(X, U[:,-1],'--k')
plt.show()

plt.figure(figsize = (20, 4))
for t__ in [0,0.2,0.4,0.6,0.8,1]:
  x_test=np.linspace(-1,1,test_resi)
  t_test = np.ones_like(x_test) * t__
  x_test, t_test = torch.tensor(x_test).to(device), torch.tensor(t_test).to(device)
  x_test_ = torch.concat([(x_test.reshape(-1,1)).float(),(t_test.reshape(-1,1)).float()],dim=1)
  # u_ = PINN_dwx.predict(x_test_).cpu().detach().numpy()
  pde_loss = PINN_dwx.cal_pdeloss(x_test_).cpu().detach().numpy()
  plt.scatter(x_test.cpu(),pde_loss)
  # plt.ylim(0,.1)

plt.show()

test_resi = 200

plt.figure(figsize = (20, 4))
for t__ in [0,0.2,0.4,0.6,0.8,1]:
  x_test=np.linspace(-1,1,test_resi)
  t_test = np.ones_like(x_test) * t__
  x_test, t_test = torch.tensor(x_test).to(device), torch.tensor(t_test).to(device)
  x_test_ = torch.concat([(x_test.reshape(-1,1)).float(),(t_test.reshape(-1,1)).float()],dim=1)
  u_ = PINN_dwo.predict(x_test_).cpu().detach().numpy()

  plt.plot(x_test.cpu(),u_,'r')
  if t__ != 1:
    plt.plot(X, U[:,int(t__*N_t)],'--k')
  else:
    plt.plot(X, U[:,-1],'--k')
  # plt.tight_layout()
  # plt.show()