import numpy as np
from typing import List, Optional
from env.models import ObservationModel, AmbulanceState, Severity

class StateEncoder:
    """
    High-quality state encoder for RL ambulance dispatch system.
    Converts ObservationModel into a fixed-size numerical vector.

    The constructor accepts an optional `max_nodes` keyword for backward
    compatibility with callers that pass StateEncoder(max_nodes=100).
    The value is ignored; MAX_NODES is always 100.
    """
    
    # Feature Configuration
    MAX_EMERGENCIES = 10
    
    # Normalization Constants
    MAX_NODES = 100
    ETA_NORM = 50
    TIME_NORM = 50
    CAPACITY_NORM = 50
    
    def __init__(self, max_nodes: int = None, max_ambulances: int = 6, max_hospitals: int = 4,
                 history_encoder=None):
        self.max_ambulances = max_ambulances
        self.max_hospitals = max_hospitals
        self.history_encoder = history_encoder  # Optional HistoryEncoder for long-horizon state
        # Deterministic mappings for one-hot encoding
        self.state_map = {
            AmbulanceState.IDLE: 0,
            AmbulanceState.DISPATCHED: 1,
            AmbulanceState.EN_ROUTE: 2,
            AmbulanceState.AT_SCENE: 3,
            AmbulanceState.TRANSPORTING: 4,
            AmbulanceState.RETURNING: 5
        }
        
        self.severity_map = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.NORMAL: 2
        }

    def encode(self, observation: ObservationModel) -> np.ndarray:
        """
        Encodes the observation model into a 1D float32 numpy array.
        """
        features = []
        
        # 1. Encode Ambulances
        for i in range(self.max_ambulances):
            if i < len(observation.ambulances):
                amb = observation.ambulances[i]
                
                # node (normalized)
                features.append(float(amb.node) / self.MAX_NODES)
                
                # state (one-hot: 6 dimensions)
                state_one_hot = [0.0] * 6
                idx = self.state_map.get(amb.state, 0)
                state_one_hot[idx] = 1.0
                features.extend(state_one_hot)
                
                # eta (normalized)
                features.append(float(amb.eta) / self.ETA_NORM)
            else:
                # Padding: 1 (node) + 6 (state) + 1 (eta) = 8 zeros
                features.extend([0.0] * 8)
        
        # 2. Encode Emergencies
        for i in range(self.MAX_EMERGENCIES):
            if i < len(observation.emergencies):
                emg = observation.emergencies[i]
                
                # node (normalized)
                features.append(float(emg.node) / self.MAX_NODES)
                
                # severity (one-hot: 3 dimensions)
                sev_one_hot = [0.0] * 3
                idx = self.severity_map.get(emg.severity, 2) # Default to NORMAL if unknown
                sev_one_hot[idx] = 1.0
                features.extend(sev_one_hot)
                
                # time_remaining (normalized)
                features.append(float(emg.time_remaining) / self.TIME_NORM)
                
                # assigned (0 or 1)
                features.append(1.0 if emg.assigned else 0.0)
            else:
                # Padding: 1 (node) + 3 (severity) + 1 (time) + 1 (assigned) = 6 zeros
                features.extend([0.0] * 6)
                
        # 3. Encode Hospitals
        for i in range(self.max_hospitals):
            if i < len(observation.hospitals):
                hosp = observation.hospitals[i]
                
                # node (normalized)
                features.append(float(hosp.node) / self.MAX_NODES)
                
                # capacity (normalized)
                features.append(float(hosp.capacity) / self.CAPACITY_NORM)
                
                # current_patients (normalized) & load ratio
                # Safe divide by capacity
                if hosp.capacity > 0:
                    patients_norm = float(hosp.current_patients) / float(hosp.capacity)
                else:
                    patients_norm = 0.0
                
                features.append(patients_norm) # current_patients (normalized)
                features.append(patients_norm) # load ratio = current / capacity
            else:
                # Padding: 1 (node) + 1 (capacity) + 1 (patients) + 1 (load_ratio) = 4 zeros
                features.extend([0.0] * 4)
                
        # Concatenate and clamp all features between 0 and 1 for stable training
        state = np.array(features, dtype=np.float32)
        state = np.clip(state, 0, 1)

        # Step 8.5 — append history features if a HistoryEncoder is attached
        if self.history_encoder is not None:
            history_features = self.history_encoder.encode()
            state = np.concatenate([state, history_features]).astype(np.float32)

        return state

    @property
    def feature_dim(self) -> int:
        """Returns the fixed size of the encoded state vector."""
        base = (self.max_ambulances * 8) + (self.MAX_EMERGENCIES * 6) + (self.max_hospitals * 4)
        if self.history_encoder is not None:
            return base + getattr(self.history_encoder, "FEATURE_DIM", 25)
        return base
