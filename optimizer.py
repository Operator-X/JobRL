from typing import List, Tuple, Dict, Any, Optional
from ortools.sat.python import cp_model

class JssOptimizer:
    """
    Classical Operations Research Solver using Google OR-Tools CP-SAT.
    Solves both classic Job Shop Scheduling (JSS) and Flexible Job Shop Scheduling (FJSP).
    """
    def __init__(self, num_jobs: int, num_machines: int, jobs_data: List[Any]):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        
        # Standardize jobs_data: List[List[List[Tuple[int, float]]]]
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

    def solve(self, time_limit_seconds: float = 30.0) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
        """
        Solves the JSS/FJSP instance.
        Returns:
            optimal_makespan: Float or None if no solution found.
            schedule_info: Dict containing start/end times and machine assignments for each job/operation.
        """
        model = cp_model.CpModel()

        # Compute horizon: upper bound of makespan
        horizon = 0
        for job_ops in self.jobs_data:
            job_max_duration = 0
            for op_options in job_ops:
                job_max_duration += max(opt[1] for opt in op_options)
            horizon += int(job_max_duration)

        # Variables: start, end, and duration for each operation
        # all_tasks[job_id, op_id] = TaskVariables
        all_tasks = {}
        # machine_intervals[machine_id] = list of interval variables processing on this machine
        machine_intervals = {m: [] for m in range(self.num_machines)}

        for job_id in range(self.num_jobs):
            job_ops = self.jobs_data[job_id]
            for op_id in range(len(job_ops)):
                op_options = job_ops[op_id]

                # Main variables for the operation
                suffix = f"_{job_id}_{op_id}"
                start_var = model.NewIntVar(0, horizon, f"start{suffix}")
                end_var = model.NewIntVar(0, horizon, f"end{suffix}")

                # Choice variables (presences) for flexible machine options
                presences = []
                opt_intervals = []

                for option_idx, (m_id, duration) in enumerate(op_options):
                    # Boolean variable: is this option chosen?
                    presence_var = model.NewBoolVar(f"presence{suffix}_{option_idx}")
                    presences.append(presence_var)

                    # Optional interval variable
                    opt_suffix = f"{suffix}_{option_idx}"
                    opt_start = model.NewIntVar(0, horizon, f"opt_start{opt_suffix}")
                    opt_end = model.NewIntVar(0, horizon, f"opt_end{opt_suffix}")
                    opt_interval = model.NewOptionalIntervalVar(
                        opt_start, int(duration), opt_end, presence_var, f"opt_interval{opt_suffix}"
                    )
                    
                    # Link main variables with option variables when present
                    model.Add(opt_start == start_var).OnlyEnforceIf(presence_var)
                    model.Add(opt_end == end_var).OnlyEnforceIf(presence_var)

                    machine_intervals[m_id].append(opt_interval)

                # Exactly one option must be chosen
                model.Add(sum(presences) == 1)

                all_tasks[job_id, op_id] = {
                    "start": start_var,
                    "end": end_var,
                    "presences": presences,
                }

        # Unary machine capacity constraints: no overlaps on any machine
        for m_id in range(self.num_machines):
            model.AddNoOverlap(machine_intervals[m_id])

        # Precedence constraints: operations within a job must be processed in order
        for job_id in range(self.num_jobs):
            job_ops = self.jobs_data[job_id]
            for op_id in range(1, len(job_ops)):
                model.Add(all_tasks[job_id, op_id]["start"] >= all_tasks[job_id, op_id - 1]["end"])

        # Objective: Minimize makespan (C_max)
        makespan = model.NewIntVar(0, horizon, "makespan")
        model.AddMaxEquality(
            makespan,
            [all_tasks[job_id, len(self.jobs_data[job_id]) - 1]["end"] for job_id in range(self.num_jobs)]
        )
        model.Minimize(makespan)

        # Solver config
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_seconds

        # Solve
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            opt_val = float(solver.ObjectiveValue())
            
            # Extract detailed schedule info
            schedule_info = {}
            for job_id in range(self.num_jobs):
                job_ops = self.jobs_data[job_id]
                for op_id in range(len(job_ops)):
                    op_options = job_ops[op_id]
                    start_time = solver.Value(all_tasks[job_id, op_id]["start"])
                    end_time = solver.Value(all_tasks[job_id, op_id]["end"])

                    # Determine selected machine
                    selected_machine = None
                    for option_idx, (m_id, duration) in enumerate(op_options):
                        if solver.BooleanValue(all_tasks[job_id, op_id]["presences"][option_idx]):
                            selected_machine = m_id
                            break

                    schedule_info[f"job_{job_id}_step_{op_id}"] = {
                        "job_id": job_id,
                        "op_idx": op_id,
                        "machine_id": selected_machine,
                        "start": float(start_time),
                        "end": float(end_time),
                    }
                    
            return opt_val, schedule_info

        return None, None
