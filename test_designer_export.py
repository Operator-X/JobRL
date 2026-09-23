import os
import unittest
from designer_core import (
    get_preset_3x3,
    get_preset_7x6,
    generate_random_scenario,
    export_to_python_code,
    export_to_json
)
from config import FactoryConfig
from engine import JssEngine
from optimizer import JssOptimizer

class TestDesignerExportIntegration(unittest.TestCase):
    """
    Simulates a team member receiving an exported scenario_parameters.py
    and running their numerical methods and DES simulation on it.
    """

    def test_handoff_preset_3x3(self):
        scenario = get_preset_3x3()
        code_str = export_to_python_code(scenario)
        
        # Execute the generated code in a clean namespace
        ns = {}
        exec(code_str, ns)
        
        self.assertIn("NUM_JOBS", ns)
        self.assertIn("NUM_MACHINES", ns)
        self.assertIn("JOBS_DATA", ns)
        self.assertIn("FACTORY_CONFIG", ns)
        
        num_jobs = ns["NUM_JOBS"]
        num_machines = ns["NUM_MACHINES"]
        jobs_data = ns["JOBS_DATA"]
        cfg = FactoryConfig(**ns["FACTORY_CONFIG"])
        
        # 1. Test DES Engine
        engine = JssEngine(num_jobs, num_machines, jobs_data, config=cfg)
        mask = engine.get_action_mask()
        self.assertTrue(any(mask))
        
        # Dispatch valid action
        first_valid = next(i for i, v in enumerate(mask) if v == 1)
        target_machine = engine.jobs[first_valid].current_op_options[0][0]
        engine.dispatch(first_valid, target_machine)
        dt, idle_times = engine.fast_forward()
        self.assertGreaterEqual(engine.current_time, 0.0)

        # 2. Test CP-SAT Optimizer
        optimizer = JssOptimizer(num_jobs, num_machines, jobs_data)
        makespan, schedule = optimizer.solve(time_limit_seconds=5.0)
        self.assertIsNotNone(makespan)
        self.assertGreater(makespan, 0.0)
        print(f"Verified 3x3 Export -> CP-SAT makespan: {makespan}")

    def test_handoff_preset_7x6(self):
        scenario = get_preset_7x6()
        code_str = export_to_python_code(scenario)
        
        ns = {}
        exec(code_str, ns)
        
        num_jobs = ns["NUM_JOBS"]
        num_machines = ns["NUM_MACHINES"]
        jobs_data = ns["JOBS_DATA"]
        cfg = FactoryConfig(**ns["FACTORY_CONFIG"])
        
        # 1. Test DES Engine
        engine = JssEngine(num_jobs, num_machines, jobs_data, config=cfg)
        mask = engine.get_action_mask()
        self.assertEqual(len(mask), 7)
        self.assertTrue(any(mask))

        # 2. Test CP-SAT Optimizer
        optimizer = JssOptimizer(num_jobs, num_machines, jobs_data)
        makespan, schedule = optimizer.solve(time_limit_seconds=5.0)
        self.assertIsNotNone(makespan)
        self.assertEqual(makespan, 28.0)
        print(f"Verified 7x6 Export -> CP-SAT makespan: {makespan}")

    def test_handoff_flexible_scenario(self):
        scenario = generate_random_scenario(
            num_jobs=4,
            num_machines=5,
            mode="flexible",
            num_groups=2,
            seed=99
        )
        code_str = export_to_python_code(scenario)
        
        ns = {}
        exec(code_str, ns)
        
        num_jobs = ns["NUM_JOBS"]
        num_machines = ns["NUM_MACHINES"]
        jobs_data = ns["JOBS_DATA"]
        cfg = FactoryConfig(**ns["FACTORY_CONFIG"])
        
        # Test Flexible DES Engine
        engine = JssEngine(num_jobs, num_machines, jobs_data, config=cfg)
        mask = engine.get_action_mask()
        self.assertEqual(len(mask), num_jobs * num_machines)
        self.assertTrue(any(mask))
        
        # Dispatch valid action in flexible mode
        valid_idx = next(i for i, v in enumerate(mask) if v == 1)
        j_id = valid_idx // num_machines
        m_id = valid_idx % num_machines
        engine.dispatch(j_id, m_id)
        dt, idle_times = engine.fast_forward()
        self.assertGreaterEqual(engine.current_time, 0.0)

if __name__ == "__main__":
    unittest.main()
