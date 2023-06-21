from src.model.PINN_dyn_weight import MLP
from src.model.PINN_vanilla import PINN
# from src.model.PINN import PINN
from src.sampling.LHS import LHS
from src.pde.Burger import Burger
from src.sampling.RAR import RAR
import torch
import numpy as np
import matplotlib.pyplot as plt


torch.manual_seed(0)
np.random.seed(0)

pts_collocation = 1000  # Number of collocation points
pts_bcic = 200  # Number of training points in each bc/ic
Adam_epochs = 15000
L_BFGS_epochs = 5000  # 20000

# Definition of BC and IC
# Note that each column of the variable "X" consists of each variable in PDE
# For example, in this case, X[:,0] indicates x and X[:,1] indicates t
bcic_func1 = lambda X: np.zeros((X.shape[0], 1))  # BC value
bcic_func2 = lambda X: np.zeros((X.shape[0], 1))  # BC value
bcic_func3 = lambda X: -np.sin(np.pi * X[:, 0]).reshape(-1, 1)  # IC value

pde = [Burger]
# bcic = [(bcic_func1, "D"), (bcic_func2, "D"), (bcic_func3, "D")]
bcic = [
	[bcic_func1, bcic_func2, bcic_func3],  #
	["D", "D", "D"]  # Type of BC/IC : "D" for Dirichlet, "N" for Neumann
]
# print(bcic[:][1])
pde_n = 1  # Number of PDE
bc_n = 2  # Number of BC (excluding IC)

# domain_bound = [ domain of x[0], domain of x[1], ... ]
domain_bound = [[-1, 1], [0, 1]]
data_collocation = LHS(domain_bound, pts_collocation)  # Collocation points sampling

# BC: u(−1, t)
data_bcic1 = np.concatenate([
	domain_bound[0][0] * np.ones((pts_bcic, 1)),  # Values of X[0]
	np.random.uniform(low=domain_bound[1][0], high=domain_bound[1][1], size=(pts_bcic, 1))  # Values of X[1]
], axis=1)

# BC: u(1, t)
data_bcic2 = np.concatenate([
	domain_bound[0][1] * np.ones((pts_bcic, 1)),  # Values of X[0]
	np.random.uniform(low=domain_bound[1][0], high=domain_bound[1][1], size=(pts_bcic, 1))  # Values of X[1]
], axis=1)

# IC: u(x, 0)
data_bcic3 = np.concatenate([
	np.random.uniform(low=domain_bound[0][0], high=domain_bound[0][1], size=(pts_bcic, 1)),  # Values of X[0]
	np.zeros((pts_bcic, 1))  # Values of X[1]
], axis=1)

# Definition of data : [pts_collocation, pts_bc1, pts_bc2,..., pts_ic1, pts_ic2,...]
data = [data_collocation, data_bcic1, data_bcic2, data_bcic3]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


net_vanilla = MLP(input_dim=2,  # For Burger, since there are x and t
                  output_dim=1,  # For Burger, since there is only u
                  hidden_layers=[25] * 4, activation='GELU')
net_vanilla = net_vanilla.to(device)
optimizer_vanilla = torch.optim.Adam(net_vanilla.parameters(), lr=0.001)
PINN_vanilla = PINN(net_vanilla, domain_bound, data, pde, bcic, bc_n, device)
PINN_vanilla.compile(optimizer_vanilla)


net_RAR = MLP(input_dim=2,  # For Burger, since there are x and t
                  output_dim=1,  # For Burger, since there is only u
                  hidden_layers=[25] * 4, activation='GELU')
net_RAR = net_RAR.to(device)
optimizer_RAR = torch.optim.Adam(net_RAR.parameters(), lr=0.001)
PINN_RAR = PINN(net_RAR, domain_bound, data, pde, bcic, bc_n, device)
PINN_RAR.compile(optimizer_RAR)

RAR_model = RAR(N=4000, m=100, sampling="Uniform")

# Train loop
# for epoch in range(Adam_epochs):
# PINN.train(epochs=Adam_epochs, history=1000)

"""
Vanilla 10000번 돌리기
"""
PINN_vanilla.train(epochs=20000, history=1000)
PINN_vanilla.train(L_BFGS=L_BFGS_epochs, history=1000)
# PINN_vanilla.train(epochs=0, history=1000, adaptive=RAR_model)

"""
Vanilla 5000번 돌리고 RAR로 1000번씩 5번 돌리기
"""
PINN_RAR.train(epochs=5000, history=1000)
PINN_RAR.train(epochs=3000, history=1000, adaptive=RAR_model)
PINN_RAR.train(epochs=3000, history=1000, adaptive=RAR_model)
PINN_RAR.train(epochs=3000, history=1000, adaptive=RAR_model)
PINN_RAR.train(epochs=3000, history=1000, adaptive=RAR_model)
PINN_RAR.train(epochs=3000, history=1000, adaptive=RAR_model)
# PINN_RAR.train(epochs=5, history=1000)
# PINN_RAR.train(epochs=1, history=1000, adaptive=RAR_model)
# PINN_RAR.train(epochs=1, history=1000, adaptive=RAR_model)
# PINN_RAR.train(epochs=1, history=1000, adaptive=RAR_model)
# PINN_RAR.train(epochs=1, history=1000, adaptive=RAR_model)
# PINN_RAR.train(epochs=1, history=1000, adaptive=RAR_model)
PINN_RAR.train(L_BFGS=L_BFGS_epochs, history=1000)



from scipy.integrate import odeint

# https://github.com/sachabinder/Burgers_equation_simulation/blob/main/Burgers_solver_SP.py
############## SET-UP THE PROBLEM ###############

mu = 1
nu = 0.01 / np.pi  # kinematic viscosity coefficient

# Spatial mesh
x_max, x_min = 1, -1  # Range of the domain according to x [m]
dx = 0.001  # Infinitesimal distance
N_x = int((x_max - x_min) / dx)  # Points number of the spatial mesh
X = np.linspace(x_min, x_max, N_x)  # Spatial array

# Temporal mesh
L_t = 1  # Duration of simulation [s]
dt = 0.001  # Infinitesimal time
N_t = int(L_t / dt)  # Points number of the temporal mesh
T = np.linspace(0, L_t, N_t)  # Temporal array

# Wave number discretization
k = 2 * np.pi * np.fft.fftfreq(N_x, d=dx)

# Def of the initial condition
u0 = -np.sin(np.pi * X)  # Single space variable fonction that represent the wave form at t = 0


# viz_tools.plot_a_frame_1D(X,u0,0,L_x,0,1.2,'Initial condition')

############## EQUATION SOLVING ###############

# Definition of ODE system (PDE ---(FFT)---> ODE system)
def burg_system(u, t, k, mu, nu):
	# Spatial derivative in the Fourier domain
	u_hat = np.fft.fft(u)
	u_hat_x = 1j * k * u_hat
	u_hat_xx = -k ** 2 * u_hat

	# Switching in the spatial domain
	u_x = np.fft.ifft(u_hat_x)
	u_xx = np.fft.ifft(u_hat_xx)

	# ODE resolution
	u_t = -mu * u * u_x + nu * u_xx
	return u_t.real


# PDE resolution (ODE system resolution)
U = odeint(burg_system, u0, T, args=(k, mu, nu,), mxstep=5000).T

test_resi = 200

plt.figure(dpi=300)
for t__ in [0, 0.2, 0.4, 0.6, 0.8, 1]:
	x_test = np.linspace(-1, 1, test_resi)
	t_test = np.ones_like(x_test) * t__
	x_test, t_test = torch.tensor(x_test).to(device), torch.tensor(t_test).to(device)
	x_test_ = torch.concat([(x_test.reshape(-1, 1)).float(), (t_test.reshape(-1, 1)).float()], dim=1)
	u_ = PINN_vanilla.predict(x_test_).cpu().detach().numpy()
	plt.plot(x_test.cpu(), u_, 'r')
	if t__ != 1:
		plt.plot(X, U[:, int(t__ * N_t)], '--k')
	else:
		plt.plot(X, U[:, -1], '--k')
plt.show()

plt.figure(dpi=300)
for t__ in [0, 0.2, 0.4, 0.6, 0.8, 1]:
	x_test = np.linspace(-1, 1, test_resi)
	t_test = np.ones_like(x_test) * t__
	x_test, t_test = torch.tensor(x_test).to(device), torch.tensor(t_test).to(device)
	x_test_ = torch.concat([(x_test.reshape(-1, 1)).float(), (t_test.reshape(-1, 1)).float()], dim=1)
	u_ = PINN_RAR.predict(x_test_).cpu().detach().numpy()
	plt.plot(x_test.cpu(), u_, 'r')
	if t__ != 1:
		plt.plot(X, U[:, int(t__ * N_t)], '--k')
	else:
		plt.plot(X, U[:, -1], '--k')
plt.show()

fig, ax = plt.subplots(dpi=300)
ax.scatter(PINN_RAR.data[0][pts_collocation:,0], PINN_RAR.data[0][pts_collocation:,1])
ax.set_xlim(domain_bound[0])
ax.set_ylim(domain_bound[1])
plt.show()

# plt.figure(figsize = (20, 4))
# for t__ in [0,0.2,0.4,0.6,0.8,1]:
#     x_test=np.linspace(-1,1,test_resi)
#     t_test = np.ones_like(x_test) * t__
#     x_test, t_test = torch.tensor(x_test).to(device), torch.tensor(t_test).to(device)
#     x_test_ = torch.concat([(x_test.reshape(-1,1)).float(),(t_test.reshape(-1,1)).float()],dim=1)
#     # u_ = PINN_dwx.predict(x_test_).cpu().detach().numpy()
#     pde_loss = PINN.cal_pde_loss(x_test_).cpu().detach().numpy()
#     plt.scatter(x_test.cpu(), pde_loss)
#     # plt.ylim(0,.1)
#
# plt.show()

# test_resi = 200
#
# plt.figure(figsize = (20, 4))
# for t__ in [0,0.2,0.4,0.6,0.8,1]:
#     x_test=np.linspace(-1,1,test_resi)
#     t_test = np.ones_like(x_test) * t__
#     x_test, t_test = torch.tensor(x_test).to(device), torch.tensor(t_test).to(device)
#     x_test_ = torch.concat([(x_test.reshape(-1,1)).float(),(t_test.reshape(-1,1)).float()],dim=1)
#     u_ = PINN.predict(x_test_).cpu().detach().numpy()
#
#     plt.plot(x_test.cpu(),u_,'r')
#     if t__ != 1:
#         plt.plot(X, U[:,int(t__*N_t)],'--k')
#     else:
#         plt.plot(X, U[:,-1],'--k')
# plt.tight_layout()
# plt.show()
