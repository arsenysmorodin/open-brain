from src.models.lif_neuron import LIFNeuron
from src.models.synapse import Synapse
import numpy as np

class MicroCircuit:
    """
    A small Spiking Neural Network (SNN) designed for temporal pattern classification.
    3 input neurons (N0, N1, N2) fully connected to 2 output neurons (A, B).
    """
    def __init__(self):
        # 1. Neuron Initialization (5 Neurons)
        self.input_neurons = [LIFNeuron(V_thresh=-50.0) for _ in range(3)] 
        self.output_neuron_A = LIFNeuron(V_thresh=-55.0)
        self.output_neuron_B = LIFNeuron(V_thresh=-55.0)
        self.output_neurons = [self.output_neuron_A, self.output_neuron_B]
        
        # 2. Synapse Initialization (6 Synapses)
        self.synapses = []
        for input_n in self.input_neurons:
            for output_n in self.output_neurons:
                # We use the fixed, asymmetric STDP for learning
                self.synapses.append(Synapse(initial_weight=0.1, eta=0.02))
        
        # Background current to ensure output neurons are sensitive
        self.background_current = 2.5 

    def update(self, input_currents, dt, dopamine_level, is_learning_phase=True):
        """Runs one time step (dt) of the entire circuit."""
        
        # --- 1. Update Input Layer ---
        input_spikes = [n.update(input_currents[i], dt) 
                        for i, n in enumerate(self.input_neurons)]

        # --- 2. Synaptic Transmission ---
        output_currents = [0.0, 0.0]
        
        # This loop calculates the total incoming current for the output layer
        for i, pre_spiked in enumerate(input_spikes):
            for j in range(len(self.output_neurons)):
                # Synapse index: i * 2 (for input N0, N1, N2) + j (for output A, B)
                syn_index = (i * 2) + j
                syn = self.synapses[syn_index]
                
                synaptic_current = syn.get_output_current(pre_spiked)
                output_currents[j] += synaptic_current

        # --- 3. Update Output Layer and Apply Gated STDP ---
        output_spikes = []
        for j, neuron in enumerate(self.output_neurons):
            
            # Total current = Synaptic current + Background support
            total_current = output_currents[j] + self.background_current
            spiked = neuron.update(total_current, dt)
            output_spikes.append(spiked)
            
            # --- STDP Application (Learning) ---
            if is_learning_phase:
                # Apply STDP to ALL incoming synapses for this output neuron
                for i, pre_spiked in enumerate(input_spikes):
                    syn_index = (i * 2) + j
                    self.synapses[syn_index].update_traces(
                        pre_spiked, 
                        spiked, 
                        dt, 
                        dopamine_level
                    )

        return output_spikes