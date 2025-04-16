import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parámetros cosmológicos
Omega_m0 = 0.3  # Densidad de materia actual
H0 = 70  # Constante de Hubble (km/s/Mpc)
z = np.linspace(0, 2, 100)  # Rango de redshift

# Ecuaciones diferenciales para phi(z) y E(z) en Brans-Dicke
def model(state, z, omega):
    phi, E = state
    # Ecuación para dphi/dz (aproximación simplificada)
    dphi_dz = -(3 / (2 * omega + 3)) * phi * (E**2 + 2 * Omega_m0 * (1 + z)**3 / phi)
    # Ecuación para dE/dz
    dE_dz = E * (1 / (2 * omega + 3)) * (omega * dphi_dz / phi + 3 * (E**2 - Omega_m0 * (1 + z)**3 / phi))
    return [dphi_dz, dE_dz]

# Condiciones iniciales: phi(0) = 1, E(0) = 1
init_conditions = [1.0, 1.0]

# Diferentes valores de omega para comparar
omegas = [100, 500, 1000]

# Graficar E(z) y phi(z)
plt.figure(figsize=(10, 6))

# Subplot para E(z)
plt.subplot(2, 1, 1)
for omega in omegas:
    sol = odeint(model, init_conditions, z, args=(omega,))
    phi = sol[:, 0]
    E = sol[:, 1]
    plt.plot(z, E, label=f'ω = {omega}')
plt.xlabel('Redshift z')
plt.ylabel('E(z) = H(z)/H0')
plt.legend()
plt.grid(True)

# Subplot para phi(z)
plt.subplot(2, 1, 2)
for omega in omegas:
    sol = odeint(model, init_conditions, z, args=(omega,))
    phi = sol[:, 0]
    plt.plot(z, phi, label=f'ω = {omega}')
plt.xlabel('Redshift z')
plt.ylabel('φ(z)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('figure1.png')