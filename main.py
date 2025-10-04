import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import quad
from scipy.linalg import eigh

V0 = 50
R0 = 1
N = 5
B = 5
angular_momentum = [i for i in range(5)]  # range gives the number of angular momentum states to consider


def potential(r):
    return V0 * r**2 if r<R0 else V0


def phi(n, r):
    """
    Returns the n-th basis function.
    """
    phi = np.sqrt(2/B) * np.sin(n*np.pi*r/B)
    return phi


def dphi(n, r):
    """
    Returns the first derivative of the n-th basis function.
    """
    return np.sqrt(2/B) * (n * np.pi / B) * np.cos(n * np.pi * r / B)


def ddphi(n, r):
    """
    Returns the second derivative of the n-th basis function.
    """
    ddphi = - (n*np.pi/B)**2 * phi(n, r)
    return ddphi

results = {}

for l in angular_momentum:
    H = np.zeros((N, N))

    for n in range(1, N+1):
        for m in range(1, N+1):
            kinetic_integrand = lambda r: phi(m,r) * -1/2 * (ddphi(n, r) + (1/r)*dphi(n,r)) * r
            potential_integrand = lambda r: phi(m,r) * (potential(r) + l**2 / (2*r**2)) * phi(n,r) * r

            kinetic_energy, _ = quad(kinetic_integrand, 0, B)
            potential_energy, _ = quad(potential_integrand, 0, B)

            H[m-1, n-1] = kinetic_energy + potential_energy
    
    plt.imshow(H, cmap='viridis') # 'viridis' is a nice color map
    plt.colorbar(label='Energy')
    plt.title(f'Hamiltonian Matrix Heatmap for l = {l}')
    plt.xlabel('Basis function index (n)')
    plt.ylabel('Basis function index (m)')
    #plt.show()

    eigenvalues, eigenvectors = eigh(H)
    results[l] = {'energies': eigenvalues, 'vectors': eigenvectors}

energy_data = {}
state_data = {}

for l in angular_momentum:
    energy_data[l] = results[l]['energies'][:]
    state_data[l] = results[l]['vectors'][:]

lowest_energies = {l: energy_data[l][0] for l in angular_momentum}
print(lowest_energies)


"""
print('\n--- Energies ---')
print(energy_data)

print('\n--- State Vectors ---')
print(state_data)
"""

df_energies = pd.DataFrame(energy_data)
df_energies.index.name = "State (n)"

#rename columns of energies to indicate angular momentum
df_energies.columns = [f'l={l}' for l in df_energies]

print('\n--- Energy Levels (Eigenvalues) ---')
print(df_energies)

column_names = [f'State {n}' for n in range(N)]
for l in angular_momentum:
    print(f'\n--- Eigenfunctions for l = {l} ---')
    eigenfunctions = state_data[l]
    df_eigenfunctions = pd.DataFrame(eigenfunctions, columns=column_names)
    df_eigenfunctions.index.name = 'Basis Function (n)'
    df_eigenfunctions.index += 1 # To make it 1-based
    #print(df_eigenfunctions)





# --- Plotting the Energy Level Diagram ---
plt.figure(figsize=(10, 7))

# Get the number of states to plot (e.g., the lowest 3)
num_states_to_plot = 3

for state_index in range(num_states_to_plot):
    # Get the energy for this state across all calculated l values
    energies_for_state = [results[l]['energies'][state_index] for l in angular_momentum]
    
    # Plot E vs l for this state
    plt.plot(angular_momentum, energies_for_state, 'o-', label=f'State n={state_index}')

plt.title('Energy Levels vs. Angular Momentum', fontsize=16)
plt.xlabel('Angular Momentum Quantum Number (l)', fontsize=12)
plt.ylabel('Energy (E) [in natural units]', fontsize=12)
plt.xticks(angular_momentum) # Ensure ticks are on integer values of l
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()
