import numpy as np

def Uniform(domain_bound, num_pts):
    samples = np.random.uniform(
        low=np.array(domain_bound)[:,0],
        high=np.array(domain_bound)[:,1],
        size=(num_pts, len(domain_bound))
    )
    return samples