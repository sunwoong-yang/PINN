from scipy.integrate import odeint
from sampling.LHS import LHS
from pyDOE import *
import numpy as np


def Burger_FFT():
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
	return U


Burger_FFT()
