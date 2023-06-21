
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



#https://github.com/lululxvi/deepxde/blob/master/deepxde/model.py
class PINN:
  def __init__(self, net, data, pde, bcic, bc_n, device):
    """
    추후 adaptive sampling할 떄 여기 안에 있는 self.data를 외부에서 갱신하는 식으로 진행하자
    """

    self.data = data # list형식으로 [pde, bc1, bc2, ..., ic1, ic2, ....] : pde에 대한 collocation은 공유되며, 각 bc와 ic에 대한 training points는 다르기 때문에 각각 정의
    self.net = net
    self.epoch_cur = 0
    self.pde = pde
    self.bcic = bcic
    self.bc_n = bc_n
    self.device = device
  #Train 메서드에 사용될 여러 준비 함수들을 여기서 정의함. (github line 337 참고)
  def compile(self, optimizer, loss="MSE"):
    self.optimizer = optimizer
    self.loss = torch.nn.MSELoss(reduction='sum') # 일단은 MSE loss만 쓰도록

  #일단 epochs는 외부에서 돌리는거로 코딩하자. 샘플링 코딩 다 되면 그때 내부에 편입시키자.
  def train_step(self, lamb=0.1, his = False, dynamic = False, L_BFGS = None):
    """
    이전에 내가 짠 코드는 bc와 ic는 고정해서 받아오고, pde collocation points는 매번 epoch마다 랜덤 갱신했음
    pde는 각 pde로 이루어진 list 형식
    """

    #Optimizer step
    if (L_BFGS is None):
      loss_final, loss_ref = self.cal_loss(lamb=lamb, dynamic = dynamic)
      self.optimizer.zero_grad()
      loss_final.backward(retain_graph=True)
      self.optimizer.step()
      self.epoch_cur += 1

    else:
      LBF_optimizer = torch.optim.LBFGS(self.net.parameters(), lr=1, max_iter=L_BFGS)
      LBF_losses = []
      def closure():
        loss_final, loss_ref = self.cal_loss(lamb=lamb, dynamic = dynamic)
        LBF_optimizer.zero_grad()
        loss_final.backward()
        LBF_losses.append(loss_final.item())
        return loss_final

      LBF_optimizer.step(closure)

    if his:
      if (L_BFGS is None):
        if not (self.epoch_cur % his):
          print(f"Epoch: {self.epoch_cur}, Loss: {loss_ref.item()}")
      else:
          loss_final, loss_ref = self.cal_loss(lamb=lamb, dynamic = dynamic)
          print(f"L-BFGS final epoch: {len(LBF_losses)}, Loss: {loss_final.item()}")

  def cal_loss(self, lamb=0.1, dynamic = False):
    # 그냥 bc, ic의 Dirichlet 처럼 값을 명시적으로 줄 때와 다른식의 loss 계산이 필요해 이렇게 따로 처리
    collocation = self.array2var(self.data[0])
    pde_target_values = torch.zeros((collocation.shape[0],1))

    loss_pde, loss_bc, loss_ic = torch.tensor(0., requires_grad=True), torch.tensor(0., requires_grad=True), torch.tensor(0., requires_grad=True)

    for pde_ in self.pde:
      loss_pde = loss_pde + self.loss(pde_(collocation, self.net(collocation)), pde_target_values.to(self.device))

    for idx, data_ in enumerate(self.data[1:]): # PDE data는 skip
      # bcic_target_values = torch.ones((data_.shape[0],1)) * torch.tensor(self.bcic[idx](data_)) # 외부에서 정의한 bcic 함수를 통한 exact (target) solutions
      bcic_target_values = torch.tensor(self.bcic[idx][0](data_)).float().to(self.device)
      if idx < self.bc_n:
        if self.bcic[idx][1] == "D":
          loss_bc = loss_bc + self.loss(self.net(torch.tensor(data_).float().to(self.device)), bcic_target_values)
        elif self.bcic[idx][1] == "N":
          input_ = self.net(torch.tensor(data_).float().to(device))
          y_x = torch.autograd.grad(y_.sum(), input_, create_graph=True)[0][:,[0]]
          loss_bc = loss_bc + self.loss(y_x, bcic_target_values)
      else:
        if self.bcic[idx][1] == "D":
          loss_ic = loss_ic + self.loss(self.net(torch.tensor(data_).float().to(self.device)), bcic_target_values)
        elif self.bcic[idx][1] == "N":
          data_ = torch.tensor(data_, requires_grad=True).float().to(self.device)
          input_ = self.net(data_)
          y_t = torch.autograd.grad(input_.sum(), data_, create_graph=True)[0][:,[1]]
          loss_ic = loss_ic + self.loss(y_t, bcic_target_values)


    #Dyamic weights
    if dynamic == False:
      self.weights_tuned = torch.tensor([1,1])
    else:
      loss_pde_theta, loss_bc_theta, loss_ic_theta = self.cal_grad(loss_pde, dynamic[0]), self.cal_grad(loss_bc, dynamic[0]), self.cal_grad(loss_ic, dynamic[0])
      if self.epoch_cur % dynamic[1] == 0 :
        if self.epoch_cur == 0 and ( dynamic[0] in ["v1","v2"]):
          self.weights_cur = torch.tensor([ loss_pde_theta / loss_bc_theta , loss_pde_theta / loss_ic_theta ])
        elif self.epoch_cur == 0 and ( dynamic[0] == "v3"):
          self.weights_cur = torch.tensor([1/3,1/3]) # 초기 weights에 따라 너무 robust하지 않아서 그냥 초기 weights를 등분으로 설정

        self.weight_prev = self.weights_cur

        if dynamic[0] in ["v1","v2"]:
          self.weights_cur = torch.tensor([ loss_pde_theta / loss_bc_theta , loss_pde_theta / loss_ic_theta ])
        elif dynamic[0] == "v3":
          self.weights_cur = (1 - self.weight_prev[0] - self.weight_prev[1]) * torch.tensor([ loss_pde_theta / loss_bc_theta , loss_pde_theta / loss_ic_theta ])
        self.weights_tuned = (1-lamb) * self.weight_prev + lamb * self.weights_cur
      else:
        pass # wieghts_cur를 갱신할필요가 없음
    # bias correction
    if (dynamic == False) or (dynamic[0] in ["v1","v2"]):
      loss_final = loss_pde + self.weights_tuned[0] * loss_bc + self.weights_tuned[1] * loss_ic # loss_pde + lambda1*loss_bc + lambda2*loss_ic
    elif dynamic[0] == "v3":
      loss_final = (1-self.weights_tuned[0]-self.weights_tuned[1])* loss_pde + self.weights_tuned[0] * loss_bc + self.weights_tuned[1] * loss_ic # loss_pde + lambda1*loss_bc + lambda2*loss_ic
    print(loss_pde, loss_bc, loss_ic)
    return loss_final, loss_pde + loss_bc + loss_ic

  def predict(self, x):
    return self.net(x)

  def cal_grad(self, x, dw_type="v1"):
    """
    input x를 self.net의 parameter에 대해 미분하여 절댓값 형태로 return
    """
    self.optimizer.zero_grad()
    x.backward(retain_graph = True) # backward를 쓸때마다 graph가 해방됨. optimizer.step을 쓰기전에 한 epoch에서 두번이상 backward쓸거면 retain_graph를 해줘야 다음에 backward를 적용할 그래프가 존재하게됨
    deriv = torch.tensor([]).to(self.device)
    for i in self.net.parameters():
      if not i.grad == None:
        deriv = torch.concat([deriv, i.grad.view(-1)])
    self.optimizer.zero_grad()

    if dw_type=="v1":
      return torch.max(torch.abs(deriv))
    elif dw_type=="v2":
      return torch.mean(torch.abs(deriv))
  def cal_pdeloss(self, test_data):
    test_data = torch.tensor(test_data, requires_grad = True)
    loss_pde, loss_bc, loss_ic = 0, 0, 0
    pde_target_values = torch.zeros((test_data.shape[0],1))
    pde_loss_f = torch.nn.MSELoss(reduction='none')
    for pde_ in self.pde:
      loss_pde += pde_loss_f(pde_(test_data, self.net(test_data)), pde_target_values.to(self.device))
    return loss_pde

  def array2var(self, tensor_input, requires_grad=True):
      return Variable(torch.from_numpy(tensor_input).float(), requires_grad=requires_grad).to(self.device)
  def tensor2var(self, tensor_input, requires_grad=True):
      return Variable(tensor_input.float(), requires_grad=requires_grad).to(self.device)

