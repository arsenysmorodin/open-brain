# simulation.py

import numpy as np
import matplotlib.pyplot as plt
from src.models.lif_neuron import LIFNeuron

# === 1. Simulation Setup ===
T = 100    # Total simulation time in ms (100 ms)
dt = 1.0   # Time step size (1 ms)
steps = int(T / dt) # Total number of steps
time = np.arange(0, T, dt)

# Input signal (current)
# Applying a constant input current of 1.8 nA to induce spiking
I_input = np.ones(steps) * 1.8 
# I_input = np.zeros(steps) # Uncomment to test resting state

# === 2. Initialization ===
neuron = LIFNeuron()
potential_history = []
spike_events = []

# === 3. Run Simulation Loop ===
for t in range(steps):
    # Pass the current input to the neuron
    is_spiking = neuron.update(I_input[t], dt)
    
    # Record the current state
    potential_history.append(neuron.potential)
    
    # Record spike times for visualization
    if is_spiking:
        spike_events.append(t * dt)

# === 4. Visualization ===
plt.figure(figsize=(10, 5))

# Plot Membrane Potential
plt.plot(time, potential_history, label='Membrane Potential V (mV)')

# Plot Threshold Line
plt.axhline(neuron.V_thresh, color='r', linestyle='--', label='Threshold V_thresh')

# Mark Spike Events
for spike_time in spike_events:
    plt.axvline(spike_time, color='orange', alpha=0.5, linestyle=':')

plt.title('Single LIF Neuron Simulation')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.legend()
plt.grid(True)
plt.show()

print(f"Simulation finished. Total spikes: {len(spike_events)}")