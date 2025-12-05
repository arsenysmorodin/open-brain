import numpy as np
import matplotlib.pyplot as plt
from src.models.lif_neuron import LIFNeuron
from src.core.neurotransmitter_system import NeurotransmitterSystem # New Import

# === 1. Simulation Setup ===
T = 300    # Total simulation time in ms (Increased to 300 ms)
dt = 1.0   # Time step size (1 ms)
steps = int(T / dt) 
time = np.arange(0, T, dt)

# Constant input current (Low enough to fire slowly at low dopamine)
I_input = np.ones(steps) * 1.5 

# === 2. Initialization ===
neuron = LIFNeuron()
# Initialize the Neurotransmitter System with a low baseline level
neuro_system = NeurotransmitterSystem(initial_dopamine=0.1) 
potential_history = []
spike_events = []
dopamine_history = [] # To record the change in dopamine over time

# === 3. Run Simulation Loop ===
for t in range(steps):
    current_time = t * dt
    
    # --- SIMULATE DOPAMINE RELEASE (The Reward Event) ---
    if current_time == 100: 
        # At 100ms, simulate a major reward hit (Dopamine injection)
        neuro_system.reward_hit(magnitude=0.7) 
        
    # --- NEUROTRANSMITTER DECAY ---
    if current_time > 100:
        neuro_system.decay(rate=0.005) # Dopamine slowly decays after the reward
    
    # Get current dopamine level
    dopamine_level = neuro_system.get_dopamine()
    
    # Pass input AND dopamine level to the neuron's update method
    is_spiking = neuron.update(I_input[t], dt, dopamine_level)
    
    # Record the current state
    potential_history.append(neuron.potential)
    dopamine_history.append(dopamine_level)
    
    # Record spike times
    if is_spiking:
        spike_events.append(current_time)

# === 4. Visualization ===
# Create two subplots: one for neuron activity, one for dopamine level
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# AXIS 1: Neuron Activity (Membrane Potential)
ax1.plot(time, potential_history, label='Membrane Potential V (mV)')
ax1.axhline(neuron.V_thresh, color='r', linestyle='--', alpha=0.5, label='Nominal Threshold')

# Mark Spike Events
for spike_time in spike_events:
    ax1.axvline(spike_time, color='orange', alpha=0.5, linestyle=':')

ax1.set_title('Single LIF Neuron Activity Modulated by Dopamine')
ax1.set_ylabel('Voltage (mV)')
ax1.grid(True)
ax1.legend(loc='upper right')

# AXIS 2: Dopamine Level
ax2.plot(time, dopamine_history, color='g', label='Dopamine Level')
ax2.axvline(100, color='b', linestyle='-', label='Reward Event @ 100ms')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Dopamine Level (Norm.)')
ax2.grid(True)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()

print(f"Simulation finished. Total spikes: {len(spike_events)}")