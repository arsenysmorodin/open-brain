class NeurotransmitterSystem:
    """
    Manages the global levels of neuromodulators (Dopamine and Adrenaline/Norepinephrine).
    """
    def __init__(self, initial_dopamine=0.1, initial_adrenaline=0.1):
        # Dopamine (Reward/Pleasure)
        self.dopamine_level = initial_dopamine
        # Adrenaline (Fear/Stress/Punishment)
        self.adrenaline_level = initial_adrenaline
        
    def get_levels(self):
        """Returns the current levels of both neuromodulators."""
        return self.dopamine_level, self.adrenaline_level

    # --- Dopamine Control ---
    def reward_hit(self, magnitude=0.7):
        """Simulates a reward event, rapidly increasing dopamine."""
        self.dopamine_level = max(0.0, min(1.0, self.dopamine_level + magnitude))
        
    # --- Adrenaline Control ---
    def punishment_hit(self, magnitude=0.7):
        """Simulates a punishment event, rapidly increasing adrenaline."""
        self.adrenaline_level = max(0.0, min(1.0, self.adrenaline_level + magnitude))

    # --- General Decay ---
    def decay(self, rate=0.005):
        """Simulates the natural decay of both neurotransmitters."""
        self.dopamine_level = max(0.0, self.dopamine_level - rate)
        self.adrenaline_level = max(0.0, self.adrenaline_level - rate)