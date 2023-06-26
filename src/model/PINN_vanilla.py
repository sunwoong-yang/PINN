import torch
import time
import copy
# https://github.com/lululxvi/deepxde/blob/master/deepxde/model.py
class PINN:
    def __init__(
            self,
            net,
            domain_bound,
            data,
            pde,
            bcic,
            bc_n,
            device,
            weights=torch.tensor([1, 1]),
    ):

        # Definition of data : [pts_collocation, pts_bc1, pts_bc2,..., pts_ic1, pts_ic2,...]
        self.data = copy.deepcopy(data)
        self.net = net
        self.domain_bound = domain_bound
        self.pde = pde
        self.bcic = bcic[0] # functions for BC/IC
        self.bcic_type = bcic[1] # Type of BC/IC
        self.bc_n = bc_n
        self.device = device
        self.epoch_cur = 0  # Current epoch
        # first component: weights for BC & second component: weights for IC
        self.weights = weights
        self.so_far_time = 0
        self.added_data = 0


    # Train 메서드에 사용될 여러 준비 함수들을 여기서 정의함. (github line 337 참고)
    def compile(self, optimizer, loss="MSE"):
        """
        Compile the PINN object with optimizer and loss
        :param optimizer: type of optimizer
        :param loss: type of loss function
        :return: None
        """
        self.optimizer = optimizer

        if loss == "MSE":
            self.loss = torch.nn.MSELoss(reduction='sum')  # For the current version, only MSE loss function

    # def train(self, epochs=1000, L_BFGS=False, adaptive=None, history=1000):
    def train(self, epochs=1000, L_BFGS=False, adaptive=None, history=False):
        """
        Train the PINN object
        :param L_BFGS: if not None, PINN object is trained by L_BFGS (max_iter is set as inputted integer)
        :param history: print the training history
        :return: None
        """

        train_start_time = time.time()

        if (adaptive is not None):
            adaptive.implement(self)  # 여기서 vanilla의 데이터가 바뀜
            self.added_data += adaptive.added_pts
        # Optimizer step
        if not L_BFGS:
            for epoch in range(1, epochs+1):
                loss_final, loss_components = self.cal_loss()
                self.optimizer.zero_grad()
                loss_final.backward(retain_graph=True)
                self.optimizer.step()

                if history:
                    if (epoch % history) == 0:
                        print(f"Epoch: {epoch}, Loss: {loss_final.item()}")
                    elif epoch == 1:
                        print(f"Initial loss: {loss_final.item()}")
        else:
            LBFGS_optimizer = torch.optim.LBFGS(self.net.parameters(), lr=1, max_iter=epochs)
            LBFGS_losses = []

            def closure():
                loss_final, loss_components = self.cal_loss()
                LBFGS_optimizer.zero_grad()
                loss_final.backward()
                LBFGS_losses.append(loss_final.item())
                return loss_final

            LBFGS_optimizer.step(closure)

            if history:
                loss_final, loss_components = self.cal_loss()
                # For L-BFGS, print only the final loss
                print(f"L-BFGS final epoch: {len(LBFGS_losses)}, Loss: {loss_final.item()}")

        self.so_far_time += time.time() - train_start_time
        print(f"So-far training time: {self.so_far_time : .0f}s")

    def cal_loss(self):

        collocation = self.array2tensor(self.data[0]) # Convert np.array to torch.autograd.Variable
        pde_target_values = torch.zeros((collocation.shape[0], 1)) # 0 is target values for all collocation points

        loss_pde = torch.tensor(0., requires_grad=True)
        loss_bc = torch.tensor(0., requires_grad=True)
        loss_ic = torch.tensor(0., requires_grad=True)

        for pde in self.pde:
            # Add loss for each PDE (w.r.t collocation points)
            loss_pde = loss_pde + self.loss(
                pde(collocation, self.net(collocation)),
                pde_target_values.to(self.device)
            )

        # Add loss for each BC/IC (w.r.t BC/IC points)
        for bcic_idx, bcic_pts in enumerate(self.data[1:]):

            # Set target values for each BC/IC points according to predefined self.bcic functions
            bcic_target_values = torch.tensor(self.bcic[bcic_idx](bcic_pts)).float().to(self.device)

            if bcic_idx < self.bc_n: # if BC
                if self.bcic_type[bcic_idx] == "D": # if bcic type is 'Dirichlet'
                    loss_bc = loss_bc + self.loss(
                        self.net(torch.tensor(bcic_pts).float().to(self.device)),
                        bcic_target_values
                    )
                elif self.bcic_type[bcic_idx] == "N": # if bcic type is 'Neumann'
                    input_ = self.net(torch.tensor(bcic_pts).float().to(self.device))
                    y_x = torch.autograd.grad(y_.sum(), input_, create_graph=True)[0][:, [0]]
                    loss_bc = loss_bc + self.loss(y_x, bcic_target_values)

            else: # if IC
                if self.bcic_type[bcic_idx] == "D":
                    loss_ic = loss_ic + self.loss(
                        self.net(torch.tensor(bcic_pts).float().to(self.device)),
                        bcic_target_values
                    )
                elif self.bcic_type[bcic_idx] == "N":
                    data_ = torch.tensor(bcic_pts, requires_grad=True).float().to(self.device)
                    input_ = self.net(bcic_pts)
                    y_t = torch.autograd.grad(input_.sum(), bcic_pts, create_graph=True)[0][:, [1]]
                    loss_ic = loss_ic + self.loss(y_t, bcic_target_values)

        # Without dynamic weights (fixed weights)
        loss_final = loss_pde + \
                     self.weights[0] * loss_bc + \
                     self.weights[1] * loss_ic

        return loss_final, (loss_pde, loss_bc, loss_ic)

    def predict(self, x):
        return self.net(x)

    def array2tensor(self, tensor_input, requires_grad=True):
        # return Variable(torch.from_numpy(tensor_input).float(), requires_grad=requires_grad).to(self.device)
        return torch.tensor(tensor_input, requires_grad=requires_grad).float().to(self.device)

    def cal_pde_loss(self, test_data):
        test_data.requires_grad_()
        pde_target_values = torch.zeros((test_data.shape[0], 1))
        pde_loss_f = torch.nn.MSELoss(reduction='none')
        pde_loss = 0
        for pde in self.pde:
            pde_loss += pde_loss_f(
                pde(test_data, self.net(test_data)),
                pde_target_values.to(self.device)
            )
        return pde_loss

