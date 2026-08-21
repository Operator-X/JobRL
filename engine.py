import heapq
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

from config import FactoryConfig

class JobState:
    """Represents the dynamic state of a job in the schedule."""
    def __init__(self, job_id: int, operations: List[List[Tuple[int, float]]]):
        self.job_id = job_id
        # list of operations, where each operation is a list of (machine_id, duration) options
        self.operations = operations
        self.current_op_idx = 0
        self.in_flight = False
        self.last_action_time = 0.0
        self.release_time = 0.0
        self.has_arrived = True

    @property
    def is_completed(self) -> bool:
        return self.current_op_idx >= len(self.operations)

    @property
    def current_op_options(self) -> Optional[List[Tuple[int, float]]]:
        if self.is_completed:
            return None
        return self.operations[self.current_op_idx]


class MachineState:
    """Represents the dynamic state of a machine in the shop floor."""
    def __init__(self, machine_id: int):
        self.machine_id = machine_id
        self.is_busy = False
        self.is_broken = False
        self.current_job_id: Optional[int] = None
        self.busy_until = 0.0
        self.total_work_scheduled = 0.0
        self.last_processed_job_id: Optional[int] = None
        self.active_event_id: Optional[int] = None


class JssEngine:
    """
    Discrete-Event Simulation (DES) Engine for Job Shop Scheduling (JSS & FJSP).
    Supports stochastic breakdowns, dynamic arrivals, setup times, and buffer limits.
    """
    def __init__(self, num_jobs: int, num_machines: int, jobs_data: List[Any], config: Optional[FactoryConfig] = None):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.config = config if config is not None else FactoryConfig()
        
        # Standardize jobs_data: List[List[List[Tuple[int, float]]]]
        # Each operation must be represented as a list of compatible machine-duration choices.
        self.jobs_data = []
        for job_ops in jobs_data:
            normalized_ops = []
            for op in job_ops:
                if isinstance(op, tuple):
                    normalized_ops.append([op])
                elif isinstance(op, list):
                    normalized_ops.append(list(op))
                else:
                    raise ValueError(f"Unknown operation format: {op}")
            self.jobs_data.append(normalized_ops)

        # Upper bounds for normalizations
        flat_durations = [
            opt[1] for job_ops in self.jobs_data for op_options in job_ops for opt in op_options
        ]
        self.max_op_duration = max(flat_durations) if flat_durations else 1.0
        self.total_duration_sum = sum(
            min(opt[1] for opt in op_options) for job_ops in self.jobs_data for op_options in job_ops
        ) if self.jobs_data else 1.0

        self.reset()

    def reset(self):
        """Resets the engine state to initial conditions."""
        self.current_time = 0.0
        self.event_queue: List[Tuple[float, int, int, str, Dict[str, Any]]] = []
        self.event_counter = 0
        self.log = []

        # Seeded random number generator
        seed_val = self.config.seed if self.config.seed is not None else 42
        self.np_rng = np.random.RandomState(seed_val)

        # Initialize states
        self.jobs = [JobState(i, self.jobs_data[i]) for i in range(self.num_jobs)]
        self.machines = [MachineState(j) for j in range(self.num_machines)]

        # Handle Dynamic Arrivals
        if self.config.enable_dynamic_arrivals:
            current_arrival = 0.0
            for i, job in enumerate(self.jobs):
                if i == 0:
                    job.release_time = 0.0
                    job.has_arrived = True
                else:
                    inter_arrival = self.np_rng.exponential(1.0 / self.config.arrival_rate_lambda)
                    current_arrival += inter_arrival
                    job.release_time = current_arrival
                    job.has_arrived = False
                    
                    # Schedule arrival event
                    self.push_event(
                        timestamp=job.release_time,
                        priority=2,  # JOB_ARRIVED = 2
                        event_type="JOB_ARRIVED",
                        payload={"job_id": job.job_id}
                    )
        else:
            for job in self.jobs:
                job.release_time = 0.0
                job.has_arrived = True

        # Handle Machine Breakdowns
        if self.config.enable_breakdowns:
            for machine in self.machines:
                mttf = self.np_rng.exponential(1.0 / self.config.failure_rate_lambda)
                self.push_event(
                    timestamp=mttf,
                    priority=3,  # MACHINE_BREAKDOWN = 3
                    event_type="MACHINE_BREAKDOWN",
                    payload={"machine_id": machine.machine_id}
                )

    def push_event(self, timestamp: float, priority: int, event_type: str, payload: Dict[str, Any]) -> int:
        """Pushes an event to the priority queue. Returns the unique event ID."""
        event_id = self.event_counter
        heapq.heappush(self.event_queue, (timestamp, priority, event_id, event_type, payload))
        self.event_counter += 1
        return event_id

    def get_action_mask(self) -> np.ndarray:
        """
        Returns a boolean array representing legal actions.
        In simplified mode, action_mask is shape (num_jobs,).
        In flexible mode, action_mask is shape (num_jobs * num_machines,).
        """
        if self.config.mode == "simplified":
            mask = np.zeros(self.num_jobs, dtype=np.int8)
            for i, job in enumerate(self.jobs):
                if job.is_completed or job.in_flight or not job.has_arrived:
                    continue
                
                # Check static required machine (first option in simplified mode)
                op_options = job.current_op_options
                if op_options:
                    machine_id, _ = op_options[0]
                    machine = self.machines[machine_id]
                    
                    # Check buffer limit on NEXT machine
                    if self.config.enable_buffer_limits and job.current_op_idx + 1 < len(job.operations):
                        next_op_options = job.operations[job.current_op_idx + 1]
                        all_next_full = True
                        for next_m_id, _ in next_op_options:
                            waiting_count = sum(
                                1 for j in self.jobs if not j.is_completed and not j.in_flight and j.has_arrived
                                and j.current_op_options and any(opt[0] == next_m_id for opt in j.current_op_options)
                            )
                            if waiting_count < self.config.buffer_capacity:
                                all_next_full = False
                                break
                        if all_next_full:
                            continue

                    if not machine.is_busy and not machine.is_broken:
                        mask[i] = 1
            return mask
            
        else:  # flexible mode
            mask = np.zeros(self.num_jobs * self.num_machines, dtype=np.int8)
            for i, job in enumerate(self.jobs):
                if job.is_completed or job.in_flight or not job.has_arrived:
                    continue
                
                op_options = job.current_op_options
                if op_options:
                    # Collect compatible machines
                    compat_machines = {m_id: dur for m_id, dur in op_options}
                    for m_id in compat_machines:
                        machine = self.machines[m_id]
                        
                        # Check buffer limit on NEXT machine
                        if self.config.enable_buffer_limits and job.current_op_idx + 1 < len(job.operations):
                            next_op_options = job.operations[job.current_op_idx + 1]
                            all_next_full = True
                            for next_m_id, _ in next_op_options:
                                waiting_count = sum(
                                    1 for j in self.jobs if not j.is_completed and not j.in_flight and j.has_arrived
                                    and j.current_op_options and any(opt[0] == next_m_id for opt in j.current_op_options)
                                )
                                if waiting_count < self.config.buffer_capacity:
                                    all_next_full = False
                                    break
                            if all_next_full:
                                continue

                        if not machine.is_busy and not machine.is_broken:
                            mask[i * self.num_machines + m_id] = 1
            return mask

    def dispatch(self, job_id: int, machine_id: int):
        """
        Dispatches Job job_id's current operation onto Machine machine_id.
        """
        job = self.jobs[job_id]
        assert not job.is_completed, f"Cannot dispatch completed job {job_id}"
        assert not job.in_flight, f"Job {job_id} is already in-flight"
        assert job.has_arrived, f"Job {job_id} has not arrived yet"

        # Find compatible option for selected machine
        op_options = job.current_op_options
        compat_option = next((opt for opt in op_options if opt[0] == machine_id), None)
        assert compat_option is not None, f"Machine {machine_id} is not compatible with Job {job_id} step {job.current_op_idx}"

        machine = self.machines[machine_id]
        assert not machine.is_busy, f"Machine {machine_id} is busy"
        assert not machine.is_broken, f"Machine {machine_id} is broken"

        duration = compat_option[1]

        # Calculate sequence-dependent setup time
        setup_time = 0.0
        if self.config.enable_setup_times and machine.last_processed_job_id is not None:
            if machine.last_processed_job_id != job_id:
                # Deterministic formula: 2.0 units of time if job type changes
                setup_time = 2.0

        total_op_duration = setup_time + duration

        # Update Job
        job.in_flight = True
        job.last_action_time = self.current_time

        # Update Machine
        machine.is_busy = True
        machine.current_job_id = job_id
        machine.total_work_scheduled += total_op_duration
        machine.busy_until = self.current_time + total_op_duration
        machine.last_processed_job_id = job_id

        # Schedule OP_COMPLETED event
        event_id = self.push_event(
            timestamp=self.current_time + total_op_duration,
            priority=1,  # OP_COMPLETED = 1
            event_type="OP_COMPLETED",
            payload={"job_id": job_id, "machine_id": machine_id}
        )
        machine.active_event_id = event_id

        # Log action dispatch
        self.log.append({
            "event": "start",
            "time": self.current_time,
            "job_id": job_id,
            "op_idx": job.current_op_idx,
            "machine_id": machine_id,
            "setup_time": setup_time,
            "duration": duration,
            "event_id": event_id
        })

    def fast_forward(self) -> Tuple[float, Dict[int, float]]:
        """
        Advances the simulation clock by popping events from the priority queue
        until a decision point is reached (at least one valid action exists) or the queue is empty.
        
        Returns:
            elapsed_time: The total simulation clock advanced during this call.
            machine_idle_times: A dict mapping machine_id to the total idle time accumulated
                                during the clock advancement.
        """
        start_time = self.current_time
        machine_idle_times = {m.machine_id: 0.0 for m in self.machines}

        while self.event_queue:
            if self.is_all_completed:
                break
            mask = self.get_action_mask()
            if np.any(mask):
                break

            timestamp, priority, event_id, event_type, payload = heapq.heappop(self.event_queue)
            dt = timestamp - self.current_time
            
            if dt > 0.0:
                # Accumulate idle time for each machine in the interval [current_time, timestamp]
                for machine in self.machines:
                    idle_in_interval = max(0.0, timestamp - max(self.current_time, machine.busy_until))
                    machine_idle_times[machine.machine_id] += idle_in_interval
                
                # Advance simulation clock
                self.current_time = timestamp

            # Process Event
            if event_type == "OP_COMPLETED":
                job_id = payload["job_id"]
                machine_id = payload["machine_id"]
                
                job = self.jobs[job_id]
                machine = self.machines[machine_id]

                # Check if this completion event is still valid (not invalidated by breakdown reschedule)
                if machine.active_event_id != event_id:
                    continue

                # Log completion
                self.log.append({
                    "event": "complete",
                    "time": self.current_time,
                    "job_id": job_id,
                    "op_idx": job.current_op_idx,
                    "machine_id": machine_id
                })

                # Update states
                job.in_flight = False
                job.current_op_idx += 1
                job.last_action_time = self.current_time

                machine.is_busy = False
                machine.current_job_id = None
                machine.busy_until = self.current_time
                machine.active_event_id = None

            elif event_type == "JOB_ARRIVED":
                job_id = payload["job_id"]
                job = self.jobs[job_id]
                job.has_arrived = True
                
                self.log.append({
                    "event": "job_arrived",
                    "time": self.current_time,
                    "job_id": job_id
                })

            elif event_type == "MACHINE_BREAKDOWN":
                machine_id = payload["machine_id"]
                machine = self.machines[machine_id]

                if machine.is_broken:
                    continue

                machine.is_broken = True
                
                # Sample repair duration (MTTR)
                mttr = self.np_rng.exponential(self.config.repair_mean_mu)
                
                self.log.append({
                    "event": "breakdown",
                    "time": self.current_time,
                    "machine_id": machine_id,
                    "duration": mttr
                })

                # Schedule Repair
                self.push_event(
                    timestamp=self.current_time + mttr,
                    priority=4,  # MACHINE_REPAIRED = 4
                    event_type="MACHINE_REPAIRED",
                    payload={"machine_id": machine_id}
                )

                if machine.is_busy:
                    # Invalidate current completion event
                    machine.active_event_id = None
                    
                    # Extend completion time by mttr
                    new_completion_time = machine.busy_until + mttr
                    machine.busy_until = new_completion_time
                    
                    # Schedule new completion event
                    new_event_id = self.push_event(
                        timestamp=new_completion_time,
                        priority=1,
                        event_type="OP_COMPLETED",
                        payload={"job_id": machine.current_job_id, "machine_id": machine_id}
                    )
                    machine.active_event_id = new_event_id
                else:
                    # Machine was idle; busy_until must cover the repair period
                    machine.busy_until = self.current_time + mttr

            elif event_type == "MACHINE_REPAIRED":
                machine_id = payload["machine_id"]
                machine = self.machines[machine_id]
                
                machine.is_broken = False
                
                self.log.append({
                    "event": "repair",
                    "time": self.current_time,
                    "machine_id": machine_id
                })

                # Schedule next failure (MTTF)
                mttf = self.np_rng.exponential(1.0 / self.config.failure_rate_lambda)
                self.push_event(
                    timestamp=self.current_time + mttf,
                    priority=3,
                    event_type="MACHINE_BREAKDOWN",
                    payload={"machine_id": machine_id}
                )

        elapsed_time = self.current_time - start_time
        return elapsed_time, machine_idle_times

    @property
    def is_all_completed(self) -> bool:
        """Returns True if all jobs have completed all operations."""
        return all(job.is_completed for job in self.jobs)
