from sampling.LHS import LHS
from pyDOE import *
import numpy as np


import time
# t0 = time.time()
# lhs(2, 2000, criterion="maximin")
# print(f"lhs time: {time.time()-t0}")

t0 = time.time()
np.random.uniform(low=[1,2], high=[3,4], size=(5000,2))
print(f"np time: {time.time()-t0}")

# asa = [[1,2],[3,4]]
# print(asa[:,0])
