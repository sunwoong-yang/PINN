# https://github.com/lululxvi/deepxde/blob/master/deepxde/model.py
from src.model.MLP import MLP

import torch
from torch.autograd import Variable


class PINN:
    def __init__(self, net, data, pde, bcic, bc_n, device):
        """
        추후 adaptive sampling할 떄 여기 안에 있는 self.data를 외부에서 갱신하는 식으로 진행하자
        """

        self.data = data  # list형식으로 [pde, bc1, bc2, ..., ic1, ic2, ....] : pde에 대한 collocation은 공유되며, 각 bc와 ic에 대한 training points는 다르기 때문에 각각 정의
        self.net = net
        self.epoch_cur = 0
        self.pde = pde
        self.bcic = bcic
        self.bc_n = bc_n
        self.device = device

    # Train 메서드에 사용될 여러 준비 함수들을 여기서 정의함. (github line 337 참고)
    def compile(self, optimizer, loss="MSE"):
        self.optimizer = optimizer
        self.loss = torch.nn.MSELoss()  # 일단은 MSE loss만 쓰도록

    # 일단 epochs는 외부에서 돌리는거로 코딩하자. 샘플링 코딩 다 되면 그때 내부에 편입시키자.
    def train_step(self, lamb=0.1, his=False, dynamic=False, L_BFGS=None):
        """
        이전에 내가 짠 코드는 bc와 ic는 고정해서 받아오고, pde collocation points는 매번 epoch마다 랜덤 갱신했음
        pde는 각 pde로 이루어진 list 형식
        """

        # Optimizer step
        if (L_BFGS is None):
            loss_final, loss_ref = self.cal_loss(lamb=lamb, dynamic=dynamic)
            self.optimizer.zero_grad()
            loss_final.backward(retain_graph=True)
            self.optimizer.step()
            self.epoch_cur += 1

        else:
            LBF_optimizer = torch.optim.LBFGS(self.net.parameters(), lr=1, max_iter=L_BFGS)
            LBF_losses = []

            def closure():
                loss_final, loss_ref = self.cal_loss(lamb=lamb, dynamic=dynamic)
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
                print(f"L-BFGS final epoch: {len(LBF_losses)}, Loss: {LBF_losses[-1]}")

    def cal_loss(self, lamb=0.1, dynamic=False):
        # 그냥 bc, ic의 Dirichlet 처럼 값을 명시적으로 줄 때와 다른식의 loss 계산이 필요해 이렇게 따로 처리
        collocation = self.array2var(self.data[0])
        pde_target_values = torch.zeros((collocation.shape[0], 1))

        loss_pde, loss_bc, loss_ic = torch.tensor(0., requires_grad=True), torch.tensor(0.,
                                                                                        requires_grad=True), torch.tensor(
            0., requires_grad=True)

        for pde_ in self.pde:
            loss_pde = loss_pde + self.loss(pde_(collocation, self.net(collocation)), pde_target_values.to(self.device))

        for idx, data_ in enumerate(self.data[1:]):  # PDE data는 skip
            bcic_target_values = torch.ones((data_.shape[0], 1)) * torch.tensor(
                self.bcic[idx](data_))  # 외부에서 정의한 bcic 함수를 통한 exact (target) solutions
            bcic_target_values = torch.tensor(self.bcic[idx](data_)).float().to(self.device)
            if idx < self.bc_n:
                loss_bc = loss_bc + self.loss(self.net(torch.tensor(data_).float().to(self.device)), bcic_target_values)
            else:
                loss_ic = loss_ic + self.loss(self.net(torch.tensor(data_).float().to(self.device)), bcic_target_values)

        loss_pde_theta, loss_bc_theta, loss_ic_theta = self.cal_grad(loss_pde), self.cal_grad(loss_bc), self.cal_grad(
            loss_ic)

        # Dyamic weights
        if dynamic:
            if self.epoch_cur == 0:
                self.weights_cur = torch.tensor([loss_pde_theta / loss_bc_theta, loss_pde_theta / loss_ic_theta])
            self.weight_prev = self.weights_cur
            self.weights_cur = torch.tensor([loss_pde_theta / loss_bc_theta, loss_pde_theta / loss_ic_theta])
            self.weights_tuned = (1 - lamb) * self.weight_prev + lamb * self.weights_cur
        else:
            self.weights_tuned = torch.tensor([1, 1])
        loss_final = loss_pde + self.weights_tuned[0] * loss_bc + self.weights_tuned[
            1] * loss_ic  # loss_pde + lambda1*loss_bc + lambda2*loss_ic

        return loss_final, loss_pde + loss_bc + loss_ic

    def predict(self, x):
        return self.net(x)

    def cal_grad(self, x):
        """
        input x를 self.net의 parameter에 대해 미분하여 절댓값 형태로 return
        """
        self.optimizer.zero_grad()
        x.backward(
            retain_graph=True)  # backward를 쓸때마다 graph가 해방됨. optimizer.step을 쓰기전에 한 epoch에서 두번이상 backward쓸거면 retain_graph를 해줘야 다음에 backward를 적용할 그래프가 존재하게됨
        deriv = torch.tensor([]).to(self.device)
        for i in self.net.parameters():
            if not i.grad == None:
                deriv = torch.concat([deriv, i.grad.view(-1)])
        self.optimizer.zero_grad()

        return torch.mean(torch.abs(deriv))

    def array2var(self, tensor_input, requires_grad=True):
        return Variable(torch.from_numpy(tensor_input).float(), requires_grad=requires_grad).to(self.device)

    def tensor2var(self, tensor_input, requires_grad=True):
        return Variable(tensor_input.float(), requires_grad=requires_grad).to(self.device)

# 얘는 기존 내 코드처럼 매 epoch마다 pde의 training points를 tensor에서 variable로 변환시킬 이유가 없어서임.
# 그리고 기존 내 코드는 bc, ic까지 variable로 변환시켰는데 이럴 이유는 없어보임 why: gradient 계산이 필요없이 그냥 network에 넣어서 output만 뽑아내면 되니까
# def sampling(domain_bound,num_pts):
#   samples = lhs(len(domain_bound), samples = num_pts, criterion = "maximin")
#   for idx in range(len(domain_bound)):
#     samples[:,idx] = samples[:,idx] * ( domain_bound[idx][1] - domain_bound[idx][0] ) + domain_bound[idx][0]
#   return samples
