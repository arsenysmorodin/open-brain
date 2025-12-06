class NeurotransmitterSystem:
    # --------------------------------------------------------------------------------
    # IMPORTANT: Introduce alpha for exponential smoothing (controls emotional inertia)
    # --------------------------------------------------------------------------------
    def __init__(self, initial_dopa=0.1, initial_adre=0.0, alpha=0.2): # <-- NEW PARAMETER
        self.dopamine_level = initial_dopa
        self.adrenaline_level = initial_adre
        self.alpha = alpha # Emotional inertia coefficient (0.0=instant, 1.0=max inertia)

    def get_levels(self):
        return self.dopamine_level, self.adrenaline_level

    def decay(self, rate=0.01):
        """Applies natural decay to both neurotransmitters."""
        self.dopamine_level = max(0.0, self.dopamine_level - rate)
        self.adrenaline_level = max(0.0, self.adrenaline_level - rate)

    def reward_hit(self, magnitude=0.8):
        """
        Increases dopamine level using exponential smoothing for inertia.
        New_Level = Old_Level * (1 - alpha) + Hit_Magnitude * alpha
        """
        new_level = self.dopamine_level * (1 - self.alpha) + magnitude * self.alpha
        self.dopamine_level = min(1.0, new_level)

    def punishment_hit(self, magnitude=0.8):
        """
        Increases adrenaline level using exponential smoothing for inertia.
        """
        new_level = self.adrenaline_level * (1 - self.alpha) + magnitude * self.alpha
        self.adrenaline_level = min(1.0, new_level)