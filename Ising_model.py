#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit
from scipy.ndimage import convolve, generate_binary_structure


# In[ ]:


#%matplotlib qt


# In[ ]:


N=50
init_random = np.random.random((N,N))
lattice_n = np.zeros((N, N))
lattice_n[init_random>=0.75] = 1
lattice_n[init_random<0.75] = -1

init_random = np.random.random((N,N))
lattice_p = np.zeros((N, N))
lattice_p[init_random>=0.25] = 1
lattice_p[init_random<0.25] = -1


# In[ ]:


plt.imshow(lattice_p, cmap='Greys')


# In[ ]:


def get_energy(lattice):
    # applies the nearest neighbours summation
    kern = generate_binary_structure(2, 1) 
    kern[1][1] = False
    arr = -lattice * convolve(lattice, kern, mode='constant', cval=0)
    return arr.sum()


# In[ ]:


@numba.njit("UniTuple(f8[:], 2)(f8[:,:], i8, f8, f8)", nopython=True, nogil=True)
def metropolis(spin_arr, times, BJ, energy):
    spin_arr = spin_arr.copy()
    net_spins = np.zeros(times-1)
    net_energy = np.zeros(times-1)
    for t in range(0,times-1):
        # 2. pick random point on array and flip spin
        x = np.random.randint(0,N)
        y = np.random.randint(0,N)
        spin_i = spin_arr[x,y] #initial spin
        spin_f = spin_i*-1 #proposed spin flip
        
        # compute change in energy
        E_i = 0
        E_f = 0
        if x>0:
            E_i += -spin_i*spin_arr[x-1,y]
            E_f += -spin_f*spin_arr[x-1,y]
        if x<N-1:
            E_i += -spin_i*spin_arr[x+1,y]
            E_f += -spin_f*spin_arr[x+1,y]
        if y>0:
            E_i += -spin_i*spin_arr[x,y-1]
            E_f += -spin_f*spin_arr[x,y-1]
        if y<N-1:
            E_i += -spin_i*spin_arr[x,y+1]
            E_f += -spin_f*spin_arr[x,y+1]
        
        # 3 / 4. change state with designated probabilities
        dE = E_f-E_i
        if (dE>0)*(np.random.random() < np.exp(-BJ*dE)):
            spin_arr[x,y]=spin_f
            energy += dE
        elif dE<=0:
            spin_arr[x,y]=spin_f
            energy += dE
            
        net_spins[t] = spin_arr.sum()
        net_energy[t] = energy
            
    return net_spins, net_energy


# In[ ]:


spins_n, energies_n = metropolis(lattice_n, 1000000, 0.7, get_energy(lattice_n))
spins_p, energies_p = metropolis(lattice_p, 1000000, 0.7, get_energy(lattice_p))


# In[ ]:


plt.figure(figsize=[12,5])
plt.plot(spins_n/N**2, label='75% of spins started negative', color='r')
plt.plot(spins_p/N**2, label='75% of spins started positive', color='k')
plt.xlabel('Algorithm Time Steps', fontsize=15)
plt.ylabel(r'Average Magnetization $(\bar{m})$', fontsize=15)
plt.suptitle(r'Time Evolution of Net Magnetization for $\beta J = 0.7$', fontsize=18)
plt.legend(loc='best', fontsize=15)
plt.grid()
plt.show()


# In[ ]:


plt.figure(figsize=[12,5])
plt.plot(energies_n, label='75% of spins started negative', color='r')
plt.plot(energies_p, label='75% of spins started positive', color='k')
plt.xlabel('Algorithm Time Steps', fontsize=15)
plt.ylabel(r'Energy $E/J$', fontsize=15)
plt.suptitle(r'Time Evolution of net Energy for $\beta J = 0.7$', fontsize=18)
plt.legend(loc='best', fontsize=15)
plt.grid()
plt.show()


# In[ ]:


def get_spin_energy(lattice, BJs):
    ms = np.zeros(len(BJs))
    ms_stds = np.zeros(len(BJs))
    E_means = np.zeros(len(BJs))
    E_stds = np.zeros(len(BJs))
    for i, bj in enumerate(BJs):
        spins, energies = metropolis(lattice, 1000000, bj, get_energy(lattice))
        ms[i] = spins[-100000:].mean()/N**2
        ms_stds[i] = spins[-100000:].std()
        E_means[i] = energies[-100000:].mean()
        E_stds[i] = energies[-100000:].std()
    return ms, ms_stds, E_means, E_stds
    
BJs = np.arange(0.1, 2, 0.05)
ms_n, ms_stds_n, E_means_n, E_stds_n = get_spin_energy(lattice_n, BJs)
ms_p, ms_stds_p, E_means_p, E_stds_p = get_spin_energy(lattice_p, BJs)


# In[ ]:


plt.figure(figsize=[14,6])
plt.plot(1/BJs, ms_n, 'o--', label='75% of spins started negative', color='r')
plt.plot(1/BJs, ms_p, 'o--', label='75% of spins started positive', color='k')
plt.xlabel(r'$\left(\frac{k}{J}\right)T$', fontsize=15)
plt.ylabel(r'Net Magnetization $(\bar{m})$', fontsize=15)
plt.legend(loc='best', fontsize=15)
plt.suptitle(r'Variation of Net Magnetization with Temperature', fontsize=18)
plt.grid()
plt.show()


# In[ ]:


plt.figure(figsize=(14,6))
plt.plot(1/BJs, E_means_n, 'o--', label='75% of spins started negative', color='r')
plt.plot(1/BJs, E_means_p, 'o--', label='75% of spins started positive', color='k')
plt.xlabel(r'$\left(\frac{k}{J}\right)T$', fontsize=15)
plt.ylabel(r'Net Energy $(\bar{E})$', fontsize=15)
plt.legend(loc='best', fontsize=15)
plt.suptitle(r'Variation of Net Energy with Temperature', fontsize=18)
plt.grid()
plt.show()


# In[ ]:


plt.figure(figsize=(14,6))
plt.plot(1/BJs, E_stds_n*BJs, 'o--',label='75% of spins started negative', color='r')
plt.plot(1/BJs, E_stds_p*BJs, 'o--', label='75% of spins started positive', color='k')
plt.xlabel(r'$\left(\frac{k}{J}\right)T$', fontsize=15)
plt.ylabel(r'$C_V / k^2$', fontsize=15)
plt.legend(loc='best', fontsize=15)
plt.suptitle(r'Variation of Specific Heat Capacity $C_v$ with Temperature', fontsize=18)
plt.grid()
plt.show()


# In[ ]:


plt.figure(figsize=(14,6))
plt.plot(1/BJs,ms_stds_n*np.sqrt(BJs), 'o--', label='75% of spins started negative', color='r')
plt.plot(1/BJs, ms_stds_p*np.sqrt(BJs), 'o--',label='75% of spins started positive', color='k')
plt.xlabel(r'$\left(\frac{k}{J}\right)T$', fontsize=15)
plt.ylabel(r' Magnetic Susecptibility $(\chi)$', fontsize=15)
plt.legend(loc='best', fontsize=15)
plt.suptitle(r'Variation of Magnetic Suceptibility $\chi$ with Temperature', fontsize=18)
plt.grid()
plt.show()

