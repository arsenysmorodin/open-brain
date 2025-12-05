# src/models/synapse.py
import numpy as np

class Synapse:
    """
    Represents a plastic connection between a pre-synaptic and a post-synaptic neuron.
    Implements the STDP (Spike-Timing-Dependent Plasticity) learning rule.
    """
    def __init__(self, initial_weight=0.1, tau_stdp=20.0, eta=0.005):
        self.weight = initial_weight # Synaptic weight (strength of the connection)
        self.tau_stdp = tau_stdp     # Time constant for the STDP time window (in ms)
        self.eta = eta               # Learning rate (how fast weights change)
        
        # Time traces: Exponentially decaying records of recent spikes.
        self.pre_trace = 0.0         
        self.post_trace = 0.0

    def update_traces(self, pre_spiked, post_spiked, dt):
        """Updates the pre- and post-synaptic time traces and applies the STDP weight change."""
        
        # 1. Decay the traces (forgetting old spikes)
        self.pre_trace *= np.exp(-dt / self.tau_stdp)
        self.post_trace *= np.exp(-dt / self.tau_stdp)

        # 2. Apply STDP (learning)
        if pre_spiked:
            # Pre-before-Post: Strengthen (LTP). Depends on how recently the post-neuron fired.
            self.weight += self.eta * self.post_trace
            self.pre_trace += 1.0 # Boost the pre-trace for future post-spikes
        
        if post_spiked:
            # Post-before-Pre: Weaken (LTD). Depends on how recently the pre-neuron fired.
            self.weight -= self.eta * self.pre_trace
            self.post_trace += 1.0 # Boost the post-trace for future pre-spikes

        # 3. Clip weights (maintain stability)
        self.weight = np.clip(self.weight, 0.0, 1.0)
        
    def get_output_current(self, pre_spiked):
        """Calculates the current injected into the post-synaptic neuron."""
        if pre_spiked:
            # Inject current proportional to the weight (5.0 is an arbitrary scaling factor)
            return self.weight * 5.0 
        return 0.0