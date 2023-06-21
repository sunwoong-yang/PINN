import torch
import numpy as np
from src.sampling.LHS import LHS
from src.sampling.Uniform import Uniform
from src.model.PINN_vanilla import PINN

import copy


class RAR(PINN):
	"""
	Residual-based Adaptive Refinement (RAR)
	ref: Yu, J., Lu, L., Meng, X., & Karniadakis, G. E. (2022). Gradient-enhanced physics-informed neural networks for forward and inverse PDE problems. Computer Methods in Applied Mechanics and Engineering, 393, 114823.
	:param N: number of points to be used for calculating pde_loss
	:param m: number of points to be added into the previous self.data
	:return: None
	"""

	# def __init__(self, PINN_model, N, m):
	def __init__(self, N, m, sampling="Uniform"):
		self.N = N
		self.m = m
		self.sampling = sampling

	def implement(self, PINN_reference):
		if self.sampling == "Uniform":
			test_data = Uniform(PINN_reference.domain_bound, self.N)
		elif self.sampling == "LHS":
			test_data = LHS(PINN_reference.domain_bound, self.N)
		else:
			print("Improper sampling technique!!")

		test_data = PINN_reference.array2tensor(test_data)
		test_pde_loss = PINN_reference.cal_pde_loss(test_data).reshape(-1)
		sort_idx = torch.argsort(test_pde_loss)[-self.m:]  # sort index of the top m pde_loss
		new_data = test_data[sort_idx].cpu().detach().numpy()  # sort top m test_data w.r.t. pde_loss
		original_data = copy.deepcopy(PINN_reference.data[0])
		updated_data = np.vstack((original_data, new_data))  # add new_data to the previous self.data
		PINN_reference.data[0] = updated_data

