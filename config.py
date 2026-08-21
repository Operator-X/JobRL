from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class FactoryConfig:
    mode: Literal["simplified", "flexible"] = "simplified"
    
    # Phase 4 Parameters
    enable_breakdowns: bool = False
    failure_rate_lambda: float = 0.005  # MTTF rate parameter (lambda for exponential distribution)
    repair_mean_mu: float = 10.0        # Mean repair duration (mean for exponential distribution)
    
    enable_dynamic_arrivals: bool = False
    arrival_rate_lambda: float = 0.02   # Lambda for Poisson inter-arrival times
    
    enable_setup_times: bool = False    # Enable sequence-dependent setup times
    enable_buffer_limits: bool = False  # Enable finite buffer constraints on machines
    buffer_capacity: int = 5            # Capacity of machine input buffer queue
    
    # Core Engine Settings
    reward_type: Literal["sparse_makespan", "dense_idle_delay"] = "dense_idle_delay"
    idle_weight_gamma: float = 1.0       # γ coefficient for dense reward shaping
    seed: Optional[int] = 42
