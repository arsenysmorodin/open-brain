import librosa
import numpy as np

class AudioAdapter:
    def __init__(self, file_path, chunk_duration_ms=50):
        self.file_path = file_path
        self.chunk_duration_ms = chunk_duration_ms
        self.y = None
        self.sr = None
        self.processed_data = []

    def load_and_process(self):
        """
        Loads audio and analyzes frequency bands for each time chunk.
        """
        # 1. Load Audio
        print(f"Loading audio file: {self.file_path}...")
        self.y, self.sr = librosa.load(self.file_path, sr=22050)
        
        # Calculate samples per chunk
        samples_per_chunk = int(self.sr * (self.chunk_duration_ms / 1000.0))
        total_chunks = len(self.y) // samples_per_chunk
        
        print(f"Processing {total_chunks} chunks of {self.chunk_duration_ms}ms each...")

        # 2. Analyze chunks
        for i in range(total_chunks):
            start = i * samples_per_chunk
            end = start + samples_per_chunk
            chunk = self.y[start:end]
            
            # Fast Fourier Transform (FFT) to get frequency spectrum
            spectrum = np.abs(np.fft.rfft(chunk))
            freqs = np.fft.rfftfreq(len(chunk), 1 / self.sr)
            
            # --- Frequency Bands Definition (in Hz) ---
            # Low (Fear): 20 - 400 Hz (Rumble, heavy bass)
            # Mid (Pleasure): 400 - 3000 Hz (Voice, most music instruments)
            # High (Fear): 3000+ Hz (Screeching, hiss)
            
            low_mask = (freqs >= 20) & (freqs < 400)
            mid_mask = (freqs >= 400) & (freqs < 3000)
            high_mask = (freqs >= 3000)
            
            # Calculate energy in each band
            low_energy = np.sum(spectrum[low_mask])
            mid_energy = np.sum(spectrum[mid_mask])
            high_energy = np.sum(spectrum[high_mask])
            
            # Normalize to current (0.0 to 5.0 nA)
            # We use log scale to mimic human hearing sensitivity
            def to_current(energy):
                if energy < 1e-3: return 0.0
                val = np.log1p(energy) * 0.5 
                return min(5.0, val) # Cap at 5.0 nA

            curr_n0 = to_current(low_energy)  # Low -> N0
            curr_n1 = to_current(mid_energy)  # Mid -> N1
            curr_n2 = to_current(high_energy) # High -> N2
            
            # Determine Dominant Emotion for Neurotransmitter System
            # Logic: Mid = Dopamine; Low/High = Adrenaline
            
            dominant_band = np.argmax([low_energy, mid_energy, high_energy])
            neuro_signal = {}
            
            if dominant_band == 1: # Mid is dominant
                neuro_signal = {'type': 'reward', 'magnitude': 0.8}
            else: # Low or High is dominant
                neuro_signal = {'type': 'punishment', 'magnitude': 0.8}

            # Store processed chunk
            self.processed_data.append({
                'chunk_id': i,
                'currents': [curr_n0, curr_n1, curr_n2],
                'neuro_signal': neuro_signal
            })
            
        return self.processed_data