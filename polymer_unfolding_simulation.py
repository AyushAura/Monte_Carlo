import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Parameters
num_simulations = 1000
num_steps = 1000
temperature = 1.0
k_B = 1.0  # Boltzmann constant

def metropolis(delta_E, T):
    """Metropolis acceptance criterion"""
    if delta_E < 0:
        return True
    else:
        return np.random.rand() < np.exp(-delta_E / (k_B * T))

def simulate_unfolding(interaction_strength, allow_residue_interaction=True):
    unfolding_time = []
    energy_profile = []

    for sim in range(num_simulations):
        energy = -0.25  # Start at weak interaction
        time = 0
        coordinates = []

        for step in range(num_steps):
            delta_E = np.random.uniform(-0.5, 0.5)
            
            # Introduce residue-residue interaction
            if allow_residue_interaction:
                delta_E += interaction_strength * np.random.uniform(0, 1)

            if metropolis(delta_E, temperature):
                energy += delta_E

            coordinates.append(step)
            energy_profile.append(energy)

            if energy < -2.5:  # Threshold for 'unfolding'
                break

            time += 1

        unfolding_time.append(time)

    return np.mean(unfolding_time), energy_profile, coordinates

# Simulations
native_time, native_energy, coord_native = simulate_unfolding(-0.1, allow_residue_interaction=False)
interacting_time, interacting_energy, coord_int = simulate_unfolding(-0.1, allow_residue_interaction=True)

# Results
print(f"Mean unfolding time without interaction: {native_time:.2f}")
print(f"Mean unfolding time with interaction: {interacting_time:.2f}")
print(f"Increase in unfolding time: {((interacting_time - native_time) / native_time) * 100:.2f}%")

# Plotting energy profile (smoothed)
plt.plot(coord_native[:len(native_energy)], native_energy[:len(coord_native)], label='Native (No interaction)', alpha=0.7)
plt.plot(coord_int[:len(interacting_energy)], interacting_energy[:len(coord_int)], label='With residue interaction', alpha=0.7)
plt.xlabel('Reaction Coordinate (Simulation Step)')
plt.ylabel('Energy')
plt.title('Free Energy Profile of Polymer Unfolding')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("free_energy_profile.png")
plt.show()
