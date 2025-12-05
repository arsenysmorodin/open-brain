class NeurotransmitterSystem:
    """
    Manages the global levels of neuromodulators (e.g., Dopamine, Serotonin).
    These global levels affect the behavior of individual neurons in the network.
    """
    def __init__(self, initial_dopamine=0.1):
        # Dopamine is crucial for reward-based learning and motivation
        self.dopamine_level = initial_dopamine
        
    def get_dopamine(self):
        """Returns the current dopamine level."""
        return self.dopamine_level

    def set_dopamine(self, level):
        """Sets a new dopamine level, capping it within a defined range (0.0 to 1.0)."""
        self.dopamine_level = max(0.0, min(1.0, level))
        
    def reward_hit(self, magnitude=0.7):
        """Simulates a reward event (e.g., finding food), rapidly increasing dopamine."""
        self.dopamine_level += magnitude
        self.set_dopamine(self.dopamine_level) # Use setter to cap the level
        
    def decay(self, rate=0.005):
        """Simulates the natural decay/reuptake of the neurotransmitter over time."""
        self.dopamine_level -= rate
        self.set_dopamine(self.dopamine_level) # Use setter to cap the level