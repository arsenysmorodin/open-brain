import numpy as np
import matplotlib.pyplot as plt
from src.core.network import MicroCircuit
from src.core.neurotransmitter_system import NeurotransmitterSystem # New Import

# --- Helper Function: Define Sound Adapter Input Patterns ---
def generate_input_pattern(pattern_type, T_ms, dt):
    """
    Simulates three types of sound frequencies mapped to input neurons:
    Low Freq -> N0, Moderate Freq -> N1, High Freq -> N2
    """
    steps = int(T_ms / dt)
    currents = np.zeros((3, steps)) 
    burst_duration = 5 
    
    # We will use N1 for Reward Learning and N2 for Punishment Learning
    
    if pattern_type == 'Moderate': # Moderate Freq (Reward Source)
        start_N1 = int(20/dt)
        end_N1 = start_N1 + burst_duration
        currents[1, start_N1:end_N1] = 5.0 # Activates N1
        
    elif pattern_type == 'High': # High Freq (Punishment Source)
        start_N2 = int(20/dt)
        end_N2 = start_N2 + burst_duration
        currents[2, start_N2:end_N2] = 5.0 # Activates N2
        
    return currents

# --- Simulation Logic ---
dt = 1.0 # ms
network = MicroCircuit()
neuro_system = NeurotransmitterSystem() # Initialize the system

# Parameters
LEARNING_CYCLES = 100
PUNISHMENT_CYCLES = 50
T_CYCLE = 50 # ms per cycle

# Data history arrays
weight_history_N1A = [] # Synapse N1 -> A (Trained for Reward)
weight_history_N2A = [] # Synapse N2 -> A (Trained for Punishment)
dopa_history = []
adre_history = []

print("--- PHASE 1: REWARD LEARNING (Moderate Freq + Dopamine) ---")

# --- PHASE 1: Learning to Associate N1 with Reward (LTP) ---
for cycle in range(LEARNING_CYCLES):
    input_currents = generate_input_pattern('Moderate', T_CYCLE, dt)
    neuro_system.reward_hit(magnitude=0.8) # High Dopamine
    neuro_system.decay(rate=0.01)

    dopa, adre = neuro_system.get_levels()

    for t in range(int(T_CYCLE / dt)):
        network.update(input_currents[:, t], dt, dopa, adre)
    
    weight_history_N1A.append(network.synapses[2].weight) # N1->A is synapse index 2
    weight_history_N2A.append(network.synapses[4].weight) # N2->A is synapse index 4
    dopa_history.append(dopa)
    adre_history.append(adre)


print("--- PHASE 2: PUNISHMENT LEARNING (High Freq + Adrenaline) ---")

# --- PHASE 2: Learning to Avoid N2 with Punishment (LTD) ---
for cycle in range(PUNISHMENT_CYCLES):
    input_currents = generate_input_pattern('High', T_CYCLE, dt)
    neuro_system.dopamine_level = 0.1 
    
    # 2. Punishment Hit
    neuro_system.punishment_hit(magnitude=0.8) # High Adrenaline
    
    # 3. Decay (поддерживает низкий уровень)
    neuro_system.decay(rate=0.01) 

    dopa, adre = neuro_system.get_levels()

    for t in range(int(T_CYCLE / dt)):
        network.update(input_currents[:, t], dt, dopa, adre)
    
    weight_history_N1A.append(network.synapses[2].weight) 
    weight_history_N2A.append(network.synapses[4].weight) 
    dopa_history.append(dopa)
    adre_history.append(adre)


# --- Visualization (Three-Panel Plot) ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
total_cycles = LEARNING_CYCLES + PUNISHMENT_CYCLES

# AXIS 1: Synaptic Weight Change (Learning Curve)
ax1.plot(range(total_cycles), weight_history_N1A, label='Weight N1 -> A (Reward Learning)', color='blue')
ax1.plot(range(total_cycles), weight_history_N2A, label='Weight N2 -> A (Punishment Learning)', color='red')
ax1.axvline(LEARNING_CYCLES, color='grey', linestyle='--', label='End of Reward Phase')
ax1.set_title('Biphasic Neuromodulation: Reward (Dopa) vs. Punishment (Adre)')
ax1.set_ylabel('Synaptic Weight (w)')
ax1.grid(True)
ax1.legend()

# AXIS 2: Dopamine Level
ax2.plot(range(total_cycles), dopa_history, color='green', label='Dopamine Level (Reward)')
ax2.axhline(network.synapses[0].dopa_gate_thresh, color='green', linestyle=':', alpha=0.5)
ax2.set_ylabel('Dopa Level (Norm.)')
ax2.grid(True)
ax2.legend()

# AXIS 3: Adrenaline Level
ax3.plot(range(total_cycles), adre_history, color='orange', label='Adrenaline Level (Punishment)')
ax3.axhline(network.synapses[0].adre_gate_thresh, color='orange', linestyle=':', alpha=0.5)
ax3.axvline(LEARNING_CYCLES, color='grey', linestyle='--')
ax3.set_xlabel('Learning Cycle')
ax3.set_ylabel('Adre Level (Norm.)')
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.show()

print("\n--- Final Learning State ---")
print(f"Weight N1 -> A (Reward Path): {network.synapses[2].weight:.4f} (Should be HIGH)")
print(f"Weight N2 -> A (Punishment Path): {network.synapses[4].weight:.4f} (Should be LOW)")