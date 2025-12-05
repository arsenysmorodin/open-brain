# src/models/lif_neuron.py

class LIFNeuron:
    """
    Implements the Leaky Integrate-and-Fire (LIF) neuron model.
    A simple, yet biologically plausible, spiking neuron model used in neuromorphic engineering.
    """
    def __init__(self, V_rest=-70.0, V_thresh=-55.0, R=10.0, tau=10.0):
        # Biological parameters (in mV and ms)
        self.V_rest = V_rest      # Resting potential (the 'empty bucket' level)
        self.V_thresh = V_thresh  # Firing threshold (the 'rim of the bucket')
        
        # Leak/Resistance parameters
        self.R = R                # Membrane resistance (influences current conversion)
        self.tau = tau            # Membrane time constant (determines leak speed)
        
        # Current state
        self.potential = V_rest   # Membrane potential (current water level)

    def update(self, I_input, dt=1.0):
        """
        Updates the neuron's potential over a single time step (dt).
        
        I_input: Incoming current from other neurons/sensors (nA).
        dt: Time step size (e.g., 1 ms).
        """
        
        # Core LIF differential equation approximation: 
        # dV/dt = -(V - V_rest)/tau + R*I_input / tau
        dV = (self.R * I_input - (self.potential - self.V_rest)) / self.tau * dt
        self.potential += dV
        
        spike = False
        
        # Check for firing
        if self.potential >= self.V_thresh:
            self.fire()
            spike = True
            
        return spike

    def fire(self):
        """Resets the potential after a spike (Action Potential)."""
        self.potential = self.V_rest