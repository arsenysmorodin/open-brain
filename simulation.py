import numpy as np
import matplotlib.pyplot as plt
from src.core.network import MicroCircuit
from src.core.neurotransmitter_system import NeurotransmitterSystem 
from src.adapters.audio_adapter import AudioAdapter # Импортируем наш новый адаптер

# --- SETUP ---
AUDIO_FILE = 'input.wav' # <--- Убедись, что файл лежит в папке!
dt = 1.0 # ms
CHUNK_DURATION = 50 # ms (один цикл обучения = 50мс звука)

# Initialize modules
adapter = AudioAdapter(AUDIO_FILE, chunk_duration_ms=CHUNK_DURATION)
audio_data = adapter.load_and_process() # Получаем данные из файла

network = MicroCircuit()
neuro_system = NeurotransmitterSystem() 

# Data history arrays
weight_history_N0A = [] # Low Freq -> A (Fear)
weight_history_N1A = [] # Mid Freq -> A (Pleasure)
weight_history_N2A = [] # High Freq -> A (Fear)
dopa_history = []
adre_history = []

print(f"--- Starting Simulation based on {AUDIO_FILE} ---")

# --- MAIN LOOP OVER AUDIO CHUNKS ---
for chunk in audio_data:
    # 1. Get Inputs from Audio
    # currents = [I_N0, I_N1, I_N2] based on real spectrum
    input_currents_magnitude = chunk['currents'] 
    
    # 2. Apply Neurotransmitters based on Audio Emotion
    signal = chunk['neuro_signal']
    
    # Reset chemicals slightly (decay) before adding new ones
    neuro_system.decay(rate=0.05)
    
    if signal['type'] == 'reward':
        # Music/Voice -> Dopamine Boost
        neuro_system.reward_hit(magnitude=signal['magnitude'])
    else:
        # Rumble/Screech -> Adrenaline Boost
        # Force Reward system OFF to avoid conflict
        neuro_system.dopamine_level = 0.05 
        neuro_system.punishment_hit(magnitude=signal['magnitude'])
        
    dopa, adre = neuro_system.get_levels()
    
    # 3. Run SNN for the duration of this chunk (e.g. 50 steps of 1ms)
    steps_per_chunk = int(CHUNK_DURATION / dt)
    
    for t in range(steps_per_chunk):
        # We assume the current is constant during this short chunk
        # Create array: [N0_curr, N1_curr, N2_curr]
        currents_now = np.array(input_currents_magnitude)
        
        # Fire spikes occur randomly based on magnitude? 
        # Or we just inject constant current for this 50ms window.
        # Let's simple injection:
        network.update(currents_now, dt, dopa, adre)
        
    # 4. Record History (Once per chunk)
    weight_history_N0A.append(network.synapses[0].weight) 
    weight_history_N1A.append(network.synapses[2].weight) 
    weight_history_N2A.append(network.synapses[4].weight) 
    dopa_history.append(dopa)
    adre_history.append(adre)

# --- Visualization ---
total_chunks = len(audio_data)
time_axis = np.linspace(0, total_chunks * CHUNK_DURATION / 1000, total_chunks) # Seconds

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# AXIS 1: Learning
ax1.plot(time_axis, weight_history_N1A, label='Mid Freq (Pleasure) Synapse', color='blue')
ax1.plot(time_axis, weight_history_N0A, label='Low Freq (Fear) Synapse', color='red')
ax1.plot(time_axis, weight_history_N2A, label='High Freq (Fear) Synapse', color='purple')
ax1.set_title(f'Real-Time Learning from Audio File: {AUDIO_FILE}')
ax1.set_ylabel('Synaptic Weight')
ax1.legend()
ax1.grid(True)

# AXIS 2: Neurotransmitters
ax2.plot(time_axis, dopa_history, color='green', label='Dopamine')
ax2.plot(time_axis, adre_history, color='orange', label='Adrenaline')
ax2.set_ylabel('Chemical Level')
ax2.legend()
ax2.grid(True)

# AXIS 3: Input Spectrum (Optional visualization of what the brain heard)
# Extract inputs for plotting
n0_in = [c['currents'][0] for c in audio_data]
n1_in = [c['currents'][1] for c in audio_data]
n2_in = [c['currents'][2] for c in audio_data]

ax3.plot(time_axis, n1_in, color='lightblue', alpha=0.6, label='Mid Input')
ax3.plot(time_axis, n0_in, color='salmon', alpha=0.6, label='Low Input')
ax3.plot(time_axis, n2_in, color='violet', alpha=0.6, label='High Input')
ax3.set_ylabel('Input Current (nA)')
ax3.set_xlabel('Time (seconds)')
ax3.legend()

plt.tight_layout()
plt.show()