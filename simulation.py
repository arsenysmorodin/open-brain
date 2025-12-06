import numpy as np
import matplotlib.pyplot as plt
from src.core.network import MicroCircuit

# --- Helper Function: Define Input Patterns (using 5ms duration fix) ---
def generate_input_pattern(pattern_name, T_ms, dt):
    steps = int(T_ms / dt)
    currents = np.zeros((3, steps)) 
    
    burst_duration = 5 # 5 ms burst duration
    
    if pattern_name == 'A':
        start_N0 = int(10/dt)
        end_N0 = start_N0 + burst_duration
        currents[0, start_N0:end_N0] = 5.0
        
        start_N1 = int(20/dt)
        end_N1 = start_N1 + burst_duration
        currents[1, start_N1:end_N1] = 5.0
        
    elif pattern_name == 'B':
        start_N2 = int(15/dt)
        end_N2 = start_N2 + burst_duration
        currents[2, start_N2:end_N2] = 5.0
        
    return currents

# --- Simulation Logic ---
dt = 1.0 # ms
network = MicroCircuit()
dopa_gate_thresh = network.synapses[0].dopa_gate_thresh

# Parameters
LEARNING_CYCLES = 100
TESTING_CYCLES = 50
T_CYCLE = 50 # ms per cycle

# Data history arrays
weight_history_A = [] 
weight_history_B = [] 
dopamine_cycle_history = [] # NEW: dopa levels history

print("--- Starting Learning Phase (Pattern A + Reward) ---")

# --- PHASE 1: LEARNING (High Dopamine, Pattern A) ---
for cycle in range(LEARNING_CYCLES):
    input_A = generate_input_pattern('A', T_CYCLE, dt)
    dopa_level = 0.8 # Reward! Gate Open

    for t in range(int(T_CYCLE / dt)):
        network.update(input_A[:, t], dt, dopa_level, is_learning_phase=True)
    
    # Record data
    weight_history_A.append(network.synapses[0].weight) 
    weight_history_B.append(network.synapses[4].weight) 
    dopamine_cycle_history.append(dopa_level) # Record dopa level for this cycle
    
# --- PHASE 2: TESTING (Low Dopamine, Pattern B) ---
print("--- Starting Testing Phase (Pattern B, No Learning) ---")

for cycle in range(TESTING_CYCLES):
    input_B = generate_input_pattern('B', T_CYCLE, dt)
    dopa_level = 0.1 # No Reward! Gate Closed

    for t in range(int(T_CYCLE / dt)):
        network.update(input_B[:, t], dt, dopa_level, is_learning_phase=True) 

    # Record data
    weight_history_A.append(network.synapses[0].weight) 
    weight_history_B.append(network.synapses[4].weight) 
    dopamine_cycle_history.append(dopa_level) # Record dopa level for this cycle


# --- Visualization (Two-Panel Plot) ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
total_cycles = LEARNING_CYCLES + TESTING_CYCLES

# AXIS 1: Synaptic Weight Change (Learning Curve)
ax1.plot(range(total_cycles), weight_history_A, label='Weight N0 -> A (Targeted)', color='blue')
ax1.plot(range(total_cycles), weight_history_B, label='Weight N2 -> A (Ignored)', color='red')
ax1.axvline(LEARNING_CYCLES, color='grey', linestyle='--', label='End of Learning Phase')
ax1.set_title('Gated STDP: Temporal Pattern Classification')
ax1.set_ylabel('Synaptic Weight (w)')
ax1.grid(True)
ax1.legend()

# AXIS 2: Dopamine Level (Gating)
ax2.plot(range(total_cycles), dopamine_cycle_history, color='green', label='Dopamine Level')
ax2.axhline(dopa_gate_thresh, color='red', linestyle=':', label='Gate Threshold')
ax2.axvline(LEARNING_CYCLES, color='grey', linestyle='--') 
ax2.set_xlabel('Learning Cycle')
ax2.set_ylabel('Dopamine Level (Norm.)')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

print("\n--- Final Network State (After Learning) ---")
print(f"Weight N0 -> A (Target Pattern Component): {network.synapses[0].weight:.4f}")
print(f"Weight N2 -> A (Ignored Pattern Component): {network.synapses[4].weight:.4f}")