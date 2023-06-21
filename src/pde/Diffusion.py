from src.model.PINN_dyn_weight import *

def pde1(x, y):

  y_x = torch.autograd.grad(y.sum(), x, create_graph=True)[0][:,[0]]
  y_x2 = torch.autograd.grad(y_x.sum(), x, create_graph=True)[0][:,[0]] # 2nd derivative (https://discuss.pytorch.org/t/second-order-derivatives-of-loss-function/71797/3)
  y_t = torch.autograd.grad(y.sum(), x, create_graph=True)[0][:,[1]]
  pde = y_t - y_x2 - torch.exp(-x[:,[1]]) * (-torch.sin(torch.pi*x[:,[0]]) + torch.pi**2 * torch.sin(torch.pi*x[:,[0]]))
  return pde

BCIC_func1 = lambda x : np.zeros((x.shape[0],1))
BCIC_func2 = lambda x : np.zeros((x.shape[0],1))
BCIC_func3 = lambda x : np.sin(np.pi * x[:,0]).reshape(-1,1)


pde = [pde1]
bcic = [(BCIC_func1,"D"), (BCIC_func2,"D"), (BCIC_func3,"D")]
pde_n, bc_n =1, 2

#data 정의 : [pde_collocation, bc1, bc2,..., ic1, ic2,...]
domain_bound = [[-1,1], [0,1]]
pts_collo = 100
pts_bcic = 30
data_collocation = sampling(domain_bound, pts_collo)
data_bcic1 = np.concatenate([domain_bound[0][0] * np.ones((pts_bcic,1)), np.random.uniform(low=domain_bound[1][0], high=domain_bound[1][1], size=(pts_bcic,1))], axis=1)
data_bcic2 = np.concatenate([domain_bound[0][1] * np.ones((pts_bcic,1)), np.random.uniform(low=domain_bound[1][0], high=domain_bound[1][1], size=(pts_bcic,1))], axis=1)
data_bcic3 = np.concatenate([np.random.uniform(low=domain_bound[0][0], high=domain_bound[0][1], size=(pts_bcic,1)), np.zeros((pts_bcic,1))], axis=1)
data = [data_collocation, data_bcic1, data_bcic2, data_bcic3]

# net = Net(inputs=2, outputs=1, hidden_layers=[32,32,32,32], activation='tanh')
# net = net.to(device)
# mse_cost_function = torch.nn.MSELoss() # Mean squared error
# optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
# PINN_dwx = PINN(net, data, pde, bcic, bc_n)
# PINN_dwx.compile(optimizer)
# epochs = 6000
# for epoch in range(epochs):
#   PINN_dwx.train_step(his=1000, dynamic = False)

net = Net(inputs=2, outputs=1, hidden_layers=[32,32,32,32], activation='tanh')
net = net.to(device)
mse_cost_function = torch.nn.MSELoss() # Mean squared error
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
PINN_dwo = PINN(net, data, pde, bcic, bc_n)
PINN_dwo.compile(optimizer)
epochs = 6000
for epoch in range(epochs):
  PINN_dwo.train_step(his=1000, dynamic = ["v1",1])

x=np.linspace(-1,1,101)
t=np.linspace(0,1,101)
x_, t_ = np.meshgrid(x,t)
test_dataset = torch.concat([torch.tensor(x_.reshape(-1,1)).float(),torch.tensor(t_.reshape(-1,1)).float()],dim=1)
z_x = PINN_dwx.predict(test_dataset).detach().numpy()
z_o = PINN_dwo.predict(test_dataset).detach().numpy()
z_exact = np.sin(np.pi*x_)*np.exp(-t_)
# print(mse_cost_function(torch.tensor(z_.reshape(101,-1)),torch.tensor(z_exact)))

print("exact")
plt.contourf(t_,x_,z_exact)
plt.show()
print("No dynamic")
plt.contourf(t_,x_,z_x.reshape(101,-1))
plt.show()
print("With dynamic")
plt.contourf(t_,x_,z_o.reshape(101,-1))
plt.show()
print("No dynamic_error")
plt.contourf(t_,x_,z_exact - z_x.reshape(101,-1), levels=np.linspace(0,0.008,40),extend='both')
plt.colorbar()
plt.show()
print("With dynamic_error")
plt.contourf(t_,x_,z_exact - z_o.reshape(101,-1), levels=np.linspace(0,0.008,40),extend='both')
plt.colorbar()
plt.show()