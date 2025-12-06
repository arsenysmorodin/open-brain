# 🧠 Biphasic Neuromodulation in a Spiking Neural Network (SNN)

## 1. Overview and Goal

This project simulates a fundamental mechanism of **emotional learning** in a minimal Spiking Neural Network (SNN). The primary objective is to demonstrate how competing neuromodulators—**Dopamine (Reward/Pleasure)** and **Adrenaline (Punishment/Fear)**—act as **gating signals** to selectively control synaptic plasticity (learning and unlearning) in real-time.

The project evolves through several phases, culminating in a model that processes and learns from the time-varying spectral content of a real-world audio file.

---

## 2. Theoretical Background

The simulation relies on the following key biological concepts:

- **Spiking Neural Networks (SNNs):** A third-generation neural network model that operates using discrete events (spikes) rather than continuous values, more closely mimicking biological neurons

[Image of biological neuron signal transmission]
.

- **Spike-Timing Dependent Plasticity (STDP):** A Hebbian learning rule where the order and timing of pre- and post-synaptic spikes determine the change in synaptic strength ($\Delta w$).
- **Gated STDP (Neuromodulation):** Synaptic plasticity is not an intrinsic property but is actively regulated by chemical messengers released based on the organism's emotional state.
  - **Dopamine (DA):** Opens the gate for **Long-Term Potentiation (LTP)**, strengthening the connection (**Reward Learning**).
  - **Adrenaline (Adre):** Opens the gate for **Long-Term Depression (LTD)**, weakening the connection (**Avoidance Learning**).

---

## 3. Project Architecture

The simulation is built with a clear separation of concerns across its source files:

| File                                  | Component                          | Description                                                                                                                                                                                              |
| :------------------------------------ | :--------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/core/network.py`                 | **MicroCircuit**                   | Defines the SNN structure (Input Neurons $\rightarrow$ Output Neuron A) and manages the overall simulation step ($\Delta t$).                                                                            |
| `src/models/neuron.py`                | **Leaky Integrate-and-Fire (LIF)** | The fundamental spiking neuron model used for computation.                                                                                                                                               |
| `src/models/synapse.py`               | **Synapse**                        | Implements the **Neuromodulated STDP** rule, where $\Delta w$ is gated by external chemical levels (Dopamine and Adrenaline).                                                                            |
| `src/core/neurotransmitter_system.py` | **NeurotransmitterSystem**         | Models the chemical environment, managing the concentration levels and implementing **Emotional Inertia** (smoothing) and decay.                                                                         |
| `src/adapters/audio_adapter.py`       | **AudioAdapter**                   | Processes real audio files, converting frequency content into current inputs ($I_{N_0}, I_{N_1}, I_{N_2}$) and generating an associated **emotional signal** (Reward or Punishment) for each time chunk. |
| `simulation.py`                       | **Main Script**                    | Runs the experiment, loads the audio, manages the learning cycle, and visualizes the results.                                                                                                            |

---

## 4. Key Experimental Phases & Results

The project demonstrated sequential advancements in control and realism:

### Phase A: Dopamine Gating (Unipolar Control)

- **Goal:** Show that learning (LTP) only occurs when Dopamine is high.
- **Result:** Synaptic weights only increased during the high-Dopamine phase, remaining flat otherwise.

### Phase B: Biphasic Neuromodulation (Reward vs. Punishment)

- **Goal:** Implement competitive control: Dopamine for LTP, Adrenaline for LTD.
- **Result:** A target synapse ($\mathbf{N_1} \rightarrow \mathbf{A}$) was successfully strengthened (LTP) in the presence of Dopamine, while a different synapse ($\mathbf{N_2} \rightarrow \mathbf{A}$) was successfully weakened (LTD) in the presence of Adrenaline, proving selective learning control.

### Phase C: Real-Time Audio Processing and Emotional Inertia

- **Goal:** Connect the SNN to a real audio file, where frequency content dynamically dictates the emotional state. Introduce inertia to stabilize learning.
- **Frequency Mapping:**
  - **Mid-Frequencies ($N_1$):** Triggers **Dopamine** (Pleasure/Reward).
  - **Low/High-Frequencies ($N_0, N_2$):** Triggers **Adrenaline** (Fear/Punishment).
- **Inertia Implementation:** The `NeurotransmitterSystem` was updated with an **exponential smoothing factor ($\alpha$ )** to prevent instantaneous switching between high Dopamine and high Adrenaline, leading to a more stable emotional state and smoother learning curves.
- **Final Result:** The SNN successfully learned a stable **Mid-Frequency $\rightarrow$ High Weight** association (it 'liked' the pleasant parts) and stable **Low/High-Frequency $\rightarrow$ Low Weight** associations (it 'disliked' the chaotic parts).

---

## 5. Usage

### Prerequisites

```bash
pip install numpy matplotlib librosa
```

### Execution

1. Place your desired audio file (e.g., .wav, .mp3) in the project root directory and name it input.wav.

2. Run the main simulation script

```bash
python simulation.py
```

The script will output a three-panel plot showing:

Synaptic Weights: The learning curve of the three input paths.

Chemical Levels: The time-varying concentration of Dopamine and Adrenaline (smoothed).

Input Current: The actual energy breakdown of the audio spectrum received by the SNN.

## 6. Conclusion

This project successfully bridges computational neuroscience with signal processing, creating a working model of emotional gating. The SNN demonstrates adaptive, context-dependent plasticity, where the ability to learn and forget is dynamically regulated by simulated internal states (neuromodulators) derived from a complex, real-world stimulus (audio).
