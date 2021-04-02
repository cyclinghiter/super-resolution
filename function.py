import numpy as np 
from scipy import special

def dipole_green_function(k0, r, dim=3):
    if dim == 2:
        G = special.hankel1(0, r)
        
    if dim == 3:  
        G = 1J / (4*np.pi) * np.exp(1J * np.abs(k0 * r)) / np.abs(r)

    G = np.nan_to_num(G)
    return G
    