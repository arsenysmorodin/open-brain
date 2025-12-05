class LIFNeuron:
    """
    Implements the Leaky Integrate-and-Fire (LIF) neuron model.
    """
    def __init__(self, V_rest=-70.0, V_thresh=-55.0, R=10.0, tau=10.0):
        # Biological parameters (in mV and ms)
        self.V_rest = V_rest      
        self.V_thresh = V_thresh  
        
        # Leak/Resistance parameters
        self.R = R                
        self.tau = tau            
        
        # Current state
        self.potential = V_rest   

    def update(self, I_input, dt=1.0, dopamine_level=0.1): # ADDED DOPAMINE INPUT
        """
        Updates the neuron's potential over a single time step (dt), 
        considering neuromodulation.
        """
        
        # --- NEUROMODULATION LOGIC ---
        # Dopamine lowers the effective threshold (makes firing easier)
        MOD_FACTOR = 5.0 # Max threshold shift (in mV) due to dopamine
        
        # The effective threshold is lowered by the Dopamine Level * MOD_FACTOR
        effective_V_thresh = self.V_thresh - (dopamine_level * MOD_FACTOR)
        
        # Core LIF differential equation approximation: 
        dV = (self.R * I_input - (self.potential - self.V_rest)) / self.tau * dt
        self.potential += dV
        
        spike = False
        
        # Check for firing using the EFFECTIVE (MODULATED) THRESHOLD
        if self.potential >= effective_V_thresh: 
            self.fire()
            spike = True
            
        return spike

    def fire(self):
        """Resets the potential after a spike (Action Potential)."""
        self.potential = self.V_rest