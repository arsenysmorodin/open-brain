import numpy as np
import matplotlib.pyplot as plt
from src.models.lif_neuron import LIFNeuron
from src.models.synapse import Synapse

# === 1. Simulation Setup ===
T = 600    # Total simulation time in ms (600 ms)
dt = 1.0   # Time step size (1 ms)
steps = int(T / dt) 
time = np.arange(0, T, dt)

# === 2. Initialization ===
pre_neuron = LIFNeuron(V_thresh=-50.0)
post_neuron = LIFNeuron(V_thresh=-55.0)
synapse = Synapse(initial_weight=0.1, eta=0.01, dopa_gate_thresh=0.4) 

# New: Background current to keep the post-neuron potential high
background_current_post = 1.6 # nA. This current alone is NOT enough to fire.

# Data history arrays
weight_history = []
dopamine_history = []

# Constant input to PRE-neuron to force regular spikes for testing STDP
pre_input = np.ones(steps) * 2.0 

# === 3. Run Simulation Loop (Demonstrating the Gate) ===
for t in range(steps):
    current_time = t * dt
    
    # --- PHASE LOGIC: Simulating Dopamine Levels ---
    dopa_level = 0.0
    if current_time < 150:
        # Phase 1: Low Dopamine (No Learning)
        dopa_level = 0.1
    elif current_time < 450:
        # Phase 2: High Dopamine (Learning Enabled - Gate Open)
        dopa_level = 0.8
    else:
        # Phase 3: Dopamine Decays (No Learning)
        dopa_level = 0.2

    # --- A. Update Pre-synaptic Neuron ---
    pre_spiked = pre_neuron.update(pre_input[t], dt)
    
    # --- B. Synaptic Transmission ---
    synaptic_current = synapse.get_output_current(pre_spiked)
    
    # --- C. Update Post-synaptic Neuron (FIX: Adding background current) ---
    # The Post-neuron now receives the synaptic signal PLUS the background current
    post_input_total = synaptic_current + background_current_post 
    post_spiked = post_neuron.update(post_input_total, dt)
    
    # --- D. Gated Learning (The Crucial Step) ---
    synapse.update_traces(pre_spiked, post_spiked, dt, dopa_level)
    
    # Record data
    weight_history.append(synapse.weight)
    dopamine_history.append(dopa_level)

# === 4. Visualization ===
# (Visualization code remains the same as previous response)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# AXIS 1: Synaptic Weight Change (Learning Curve)
ax1.plot(time, weight_history, color='blue', label='Synaptic Weight w')
ax1.axvline(150, color='grey', linestyle='--', alpha=0.7)
ax1.axvline(450, color='grey', linestyle='--', alpha=0.7)
ax1.text(75, np.max(weight_history) * 1.05, 'Phase 1: No Learning', ha='center')
ax1.text(300, np.max(weight_history) * 1.05, 'Phase 2: Learning Enabled (High Dopa)', ha='center')
ax1.text(525, np.max(weight_history) * 1.05, 'Phase 3: No Learning', ha='center')
ax1.set_title('Gated STDP: Learning Controlled by Dopamine Level')
ax1.set_ylabel('Synaptic Weight (w)')
ax1.grid(True)
ax1.legend(loc='lower right')

# AXIS 2: Dopamine Level
ax2.plot(time, dopamine_history, color='g', label='Dopamine Level')
ax2.axhline(synapse.dopa_gate_thresh, color='r', linestyle=':', label='Dopamine Gate Threshold')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Dopamine Level (Norm.)')
ax2.grid(True)
ax2.legend(loc='lower right')

plt.tight_layout()
plt.show()

print(f"Simulation fixed and finished. Final Synaptic Weight: {synapse.weight:.4f}")