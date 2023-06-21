from src.model.PINN_dyn_weight import *

def pde1(x, y):

  y_x = torch.autograd.grad(y.sum(), x, create_graph=True)[0][:,[0]]
  y_x2 = torch.autograd.grad(y_x.sum(), x, create_graph=True)[0][:,[0]] # 2nd derivative (https://discuss.pytorch.org/t/second-order-derivatives-of-loss-function/71797/3)
  y_t = torch.autograd.grad(y.sum(), x, create_graph=True)[0][:,[1]]
  y_t2 = torch.autograd.grad(y_t.sum(), x, create_graph=True)[0][:,[1]]
  pde = y_t2 - 1 * y_x2
  return pde

BCIC_func1 = lambda x : np.zeros((x.shape[0],1))
BCIC_func2 = lambda x : np.zeros((x.shape[0],1))
BCIC_func3 = lambda x : 0.5 * np.sin(np.pi * x[:,0]).reshape(-1,1)
BCIC_func4 = lambda x : np.pi * np.sin(3 * np.pi * x[:,0]).reshape(-1,1)

pde = [pde1]
bcic = [(BCIC_func1,"D"), (BCIC_func2,"D"), (BCIC_func3,"D"), (BCIC_func4,"N")]
bc_n =2

#data 정의 : [pde_collocation, bc1, bc2,..., ic1, ic2,...]
domain_bound = [[0,1], [0,1]]
pts_collo = 2000
pts_bcic = 200
data_collocation = sampling(domain_bound, pts_collo)
data_bcic1 = np.concatenate([domain_bound[0][0] * np.ones((pts_bcic,1)), np.random.uniform(low=domain_bound[1][0], high=domain_bound[1][1], size=(pts_bcic,1))], axis=1)
data_bcic2 = np.concatenate([domain_bound[0][1] * np.ones((pts_bcic,1)), np.random.uniform(low=domain_bound[1][0], high=domain_bound[1][1], size=(pts_bcic,1))], axis=1)
data_bcic3 = np.concatenate([np.random.uniform(low=domain_bound[0][0], high=domain_bound[0][1], size=(pts_bcic,1)), np.zeros((pts_bcic,1))], axis=1)
data_bcic4 = np.concatenate([np.random.uniform(low=domain_bound[0][0], high=domain_bound[0][1], size=(pts_bcic,1)), np.zeros((pts_bcic,1))], axis=1)
data = [data_collocation, data_bcic1, data_bcic2, data_bcic3, data_bcic4]

# net = Net(inputs=2, outputs=1, hidden_layers=[64]*6, activation='tanh')
# net = net.to(device)
# mse_cost_function = torch.nn.MSELoss() # Mean squared error
# optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
# PINN_dwx = PINN(net, data, pde, bcic, bc_n)
# PINN_dwx.compile(optimizer)
# epochs = 15000
# for epoch in range(epochs):
#   PINN_dwx.train_step(his=1000, dynamic = False)
# PINN_dwx.train_step(his=True, dynamic=False, L_BFGS=1000)

net = Net(inputs=2, outputs=1, hidden_layers=[64]*6, activation='tanh')
net = net.to(device)
mse_cost_function = torch.nn.MSELoss() # Mean squared error
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
PINN_dwo = PINN(net, data, pde, bcic, bc_n)
PINN_dwo.compile(optimizer)
epochs = 15000
for epoch in range(epochs):
  PINN_dwo.train_step(his=1000, dynamic = ["v1",1])
# PINN_dwo.train_step(his=True, dynamic = ["v1",1], L_BFGS=1000)

x=np.linspace(domain_bound[0][0],domain_bound[0][1],101)
t=np.linspace(domain_bound[1][0],domain_bound[1][1],101)
x_, t_ = np.meshgrid(x,t)
test_dataset = torch.concat([torch.tensor(x_.reshape(-1,1)).float(),torch.tensor(t_.reshape(-1,1)).float()],dim=1).to(device)
# z_x = PINN_dwx.predict(test_dataset).detach().cpu().numpy()
z_o = PINN_dwo.predict(test_dataset).detach().cpu().numpy()
z_exact = 1/2 * np.sin(np.pi * x_) * np.cos(np.pi * t_) + 1/3 * np.sin(3 * np.pi * x_) * np.sin(3 * np.pi * t_)
# print(mse_cost_function(torch.tensor(z_.reshape(101,-1)),torch.tensor(z_exact)))

print("exact")
plt.contourf(t_,x_,z_exact)
plt.show()
# print("No dynamic")
# plt.contourf(t_,x_,z_x.reshape(101,-1))
# plt.show()
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

test_resi = 50

for t__ in [0,0.25,0.5,0.75,1,1.25]:
  x_test = np.linspace(0,1,test_resi)
  u_exact = 1/2 * np.sin(np.pi * x_test) * np.cos(np.pi * t__) + 1/3 * np.sin(3 * np.pi * x_test) * np.sin(3 * np.pi * t__)
  t_test = np.ones_like(x_test) * t__
  x_test, t_test = torch.tensor(x_test).to(device), torch.tensor(t_test).to(device)
  test_data = torch.concat([torch.tensor(x_test.reshape(-1,1)).float(),torch.tensor(t_test.reshape(-1,1)).float()],dim=1)
  u_ = PINN_dwx.predict(test_data).cpu().detach().numpy()

  plt.scatter(x_test.cpu(), u_exact, color='k', marker='o')
  plt.plot(x_test.cpu(), u_, 'r')
  plt.scatter(x_test.cpu(), u_exact)
  plt.show()

test_dataset.shape