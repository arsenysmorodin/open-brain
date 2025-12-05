import numpy as np
import matplotlib.pyplot as plt
from src.models.lif_neuron import LIFNeuron
from src.models.synapse import Synapse # New Import

# === 1. Simulation Setup ===
T = 500    # Total simulation time in ms
dt = 1.0   # Time step size (1 ms)
steps = int(T / dt) 
time = np.arange(0, T, dt)

# === 2. Initialization ===
pre_neuron = LIFNeuron(V_thresh=-55.0)  # Neuron that sends the signal
post_neuron = LIFNeuron(V_thresh=-60.0) # Neuron that receives the signal (easier to fire)
synapse = Synapse(initial_weight=0.1)   # Start with a very weak connection

# Data history arrays
weight_history = []
post_potential_history = []

# Input for the PRE-synaptic neuron (random external input)
# Simulate random noise to create occasional, timed spikes for learning
pre_input = np.random.normal(loc=1.5, scale=0.5, size=steps) 

# === 3. Run Simulation Loop (STDP Learning) ===
for t in range(steps):
    
    # --- A. Update Pre-synaptic Neuron ---
    pre_spiked = pre_neuron.update(pre_input[t], dt)
    
    # --- B. Synaptic Transmission ---
    # Synapse calculates the current based on pre-spike and weight
    synaptic_current = synapse.get_output_current(pre_spiked)
    
    # --- C. Update Post-synaptic Neuron ---
    # Post-neuron receives the synaptic current from the synapse
    post_spiked = post_neuron.update(synaptic_current, dt)
    
    # --- D. Learning (STDP) ---
    # The weight changes based on the spike timing of the two neurons
    synapse.update_traces(pre_spiked, post_spiked, dt)
    
    # Record data
    weight_history.append(synapse.weight)
    post_potential_history.append(post_neuron.potential)

# === 4. Visualization ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# AXIS 1: Post-Synaptic Neuron Activity
ax1.plot(time, post_potential_history, label='Post-synaptic Potential (mV)')
ax1.axhline(post_neuron.V_thresh, color='r', linestyle='--', alpha=0.7, label='Threshold')
ax1.set_title('Synaptic Plasticity (STDP) Demonstration: Learning a Connection')
ax1.set_ylabel('Voltage (mV)')
ax1.grid(True)
ax1.legend(loc='upper right')

# AXIS 2: Synaptic Weight Change (Learning)
ax2.plot(time, weight_history, color='g', label='Synaptic Weight w')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Weight (w)')
ax2.grid(True)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()

print(f"Simulation finished. Final Synaptic Weight: {synapse.weight:.4f}")