import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Parámetros cosmológicos
Omega_m0 = 0.3  # Densidad de materia actual
H0 = 70  # Constante de Hubble (km/s/Mpc)
z = np.linspace(0.01, 2, 100)  # Rango de redshift

# Modelo simplificado de E(z) para Brans-Dicke (aproximación para omega grande)
def E(z, omega):
    phi = 1.0  # Aproximación: phi ~ constante para omega grande
    return np.sqrt(Omega_m0 * (1 + z)**3 / phi + (1 - Omega_m0))  # Incluye término de energía oscura

# Calcular la distancia lumínica d_L(z) y el módulo de distancia μ(z)
def mu(z, omega):
    c = 3e5  # Velocidad de la luz (km/s)
    integrand = lambda zp: 1 / E(zp, omega)
    d_L = (c / H0) * (1 + z) * np.array([quad(integrand, 0, zi)[0] for zi in z])
    return 5 * np.log10(d_L) + 25  # Fórmula del módulo de distancia

# Generar datos simulados de supernovas tipo Ia (aproximación simple)
np.random.seed(42)
z_data = np.linspace(0.1, 2, 50)
mu_data = mu(z_data, 1000) + np.random.normal(0, 0.2, len(z_data))  # Añadir ruido

# Graficar μ(z) para diferentes valores de omega
omegas = [100, 500, 1000]
plt.figure(figsize=(8, 6))
for omega in omegas:
    plt.plot(z, mu(z, omega), label=f'ω = {omega}')
plt.scatter(z_data, mu_data, color='black', s=10, label='Simulated SN Ia data')
plt.xlabel('Redshift z')
plt.ylabel('Distance modulus μ(z)')
plt.legend()
plt.grid(True)
plt.savefig('figure2.png')