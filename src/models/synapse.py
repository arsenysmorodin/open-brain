import numpy as np

class Synapse:
    """
    Represents a plastic connection (Synapse) between two neurons.
    Implements the STDP learning rule, gated by a neuromodulator (Dopamine).
    """
    def __init__(self, initial_weight=0.1, tau_stdp=20.0, eta=0.005, dopa_gate_thresh=0.4):
        self.weight = initial_weight
        self.tau_stdp = tau_stdp    
        self.eta = eta              
        self.dopa_gate_thresh = dopa_gate_thresh # New: Dopamine threshold to enable learning
        
        self.pre_trace = 0.0         
        self.post_trace = 0.0

    def get_output_current(self, pre_spiked):
        """Calculates the current injected into the post-synaptic neuron."""
        if pre_spiked:
            return self.weight * 5.0 
        return 0.0
         
    def update_traces(self, pre_spiked, post_spiked, dt, dopamine_level):
        """
        Updates traces and applies GATED STDP: learning is only enabled if dopamine 
        is above the defined threshold.
        """  
        
        # 1. Decay the traces
        self.pre_trace *= np.exp(-dt / self.tau_stdp)
        self.post_trace *= np.exp(-dt / self.tau_stdp)

        # 2. BOOST TRACES (Register the spikes in the current time step) - CRITICAL FIX
        if pre_spiked:
            self.pre_trace += 1.0 
        if post_spiked:
            self.post_trace += 1.0 

        is_learning_enabled = (dopamine_level >= self.dopa_gate_thresh)
        
        # 3. Apply STDP (Learning) ONLY if the gate is open
        if is_learning_enabled:
            
            # --- Potentiation (Pre-before-Post) ---
            if pre_spiked:
                # Weight change now uses the freshly boosted self.post_trace
                self.weight += self.eta * self.post_trace * 5.0 
            
            # --- Depression (Post-before-Pre) ---
            if post_spiked:
                # Weight change now uses the freshly boosted self.pre_trace
                self.weight -= self.eta * self.pre_trace * 1.0

        # 4. Clip weights
        self.weight = np.clip(self.weight, 0.0, 1.0)
        """
        Updates traces and applies GATED STDP: learning is only enabled if dopamine 
        is above the defined threshold.
        """
        
        # 1. Decay the traces (forgetting old spikes)
        self.pre_trace *= np.exp(-dt / self.tau_stdp)
        self.post_trace *= np.exp(-dt / self.tau_stdp)

        # 2. Check the Dopamine Gate
        is_learning_enabled = (dopamine_level >= self.dopa_gate_thresh)
        
        # 3. Apply STDP (Learning) ONLY if the gate is open
        if is_learning_enabled:
            
            # --- Potentiation (Pre-before-Post) ---
            if pre_spiked:
                # The strength of learning is proportional to the post-trace (recency of post-spike)
                self.weight += self.eta * self.post_trace * 5.0
                self.pre_trace += 1.0 # Boost the pre-trace
            
            # --- Depression (Post-before-Pre) ---
            if post_spiked:
                # The strength of un-learning is proportional to the pre-trace
                self.weight -= self.eta * self.pre_trace * 1.0
                self.post_trace += 1.0 # Boost the post-trace

        # 4. Clip weights (maintain stability)
        self.weight = np.clip(self.weight, 0.0, 1.0)