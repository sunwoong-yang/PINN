import torch


def Burger(x, y):
    mu = 0.01 / torch.pi
    y_x = torch.autograd.grad(y.sum(), x, create_graph=True)[0][:, [0]]
    # 2nd derivative (https://discuss.pytorch.org/t/second-order-derivatives-of-loss-function/71797/3)
    y_x2 = torch.autograd.grad(y_x.sum(), x, create_graph=True)[0][:, [0]]
    y_t = torch.autograd.grad(y.sum(), x, create_graph=True)[0][:, [1]]
    pde = y_t + y * y_x - mu * y_x2
    return pde
