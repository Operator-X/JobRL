import unittest
from designer_core import (
    get_preset_3x3,
    get_preset_7x6,
    generate_random_scenario,
    to_jobs_data,
    validate_scenario,
    compute_workload_distribution,
    export_to_python_code,
    export_to_json,
    ScenarioDefinition,
    JobDefinition,
    OperationStep,
    MachineGroup
)
from config import FactoryConfig
from engine import JssEngine

class TestDesignerCore(unittest.TestCase):
    
    def test_preset_3x3(self):
        scenario = get_preset_3x3()
        is_valid, errors, warnings = validate_scenario(scenario)
        self.assertTrue(is_valid, f"Preset 3x3 validation failed: {errors}")
        
        jobs_data = to_jobs_data(scenario)
        self.assertEqual(len(jobs_data), 3)
        self.assertEqual(len(jobs_data[0]), 3)
        
        # Test directly in JssEngine
        engine = JssEngine(scenario.num_jobs, scenario.num_machines, jobs_data)
        mask = engine.get_action_mask()
        self.assertEqual(len(mask), 3)

    def test_preset_7x6(self):
        scenario = get_preset_7x6()
        is_valid, errors, warnings = validate_scenario(scenario)
        self.assertTrue(is_valid, f"Preset 7x6 validation failed: {errors}")
        
        jobs_data = to_jobs_data(scenario)
        self.assertEqual(len(jobs_data), 7)
        self.assertEqual(len(jobs_data[0]), 6)
        
        # Test directly in JssEngine
        engine = JssEngine(scenario.num_jobs, scenario.num_machines, jobs_data)
        mask = engine.get_action_mask()
        self.assertEqual(len(mask), 7)

    def test_random_simplified_generator(self):
        scenario = generate_random_scenario(num_jobs=5, num_machines=4, mode="simplified", seed=42)
        is_valid, errors, warnings = validate_scenario(scenario)
        self.assertTrue(is_valid, f"Random simplified validation failed: {errors}")
        
        jobs_data = to_jobs_data(scenario)
        engine = JssEngine(5, 4, jobs_data)
        mask = engine.get_action_mask()
        self.assertTrue(any(mask))

    def test_random_flexible_generator(self):
        scenario = generate_random_scenario(
            num_jobs=4,
            num_machines=6,
            mode="flexible",
            num_groups=3,
            seed=123
        )
        is_valid, errors, warnings = validate_scenario(scenario)
        self.assertTrue(is_valid, f"Random flexible validation failed: {errors}")
        
        jobs_data = to_jobs_data(scenario)
        cfg = FactoryConfig(mode="flexible")
        engine = JssEngine(4, 6, jobs_data, config=cfg)
        mask = engine.get_action_mask()
        # In flexible mode, mask shape is (num_jobs * num_machines)
        self.assertEqual(len(mask), 4 * 6)
        self.assertTrue(any(mask))

    def test_workload_distribution(self):
        scenario = get_preset_3x3()
        stats = compute_workload_distribution(scenario)
        self.assertIn("machine_workload", stats)
        self.assertIn("group_workload", stats)
        self.assertIn("total_workload", stats)
        self.assertIn("bottleneck_machine", stats)
        self.assertGreater(stats["total_workload"], 0)

    def test_code_exporters(self):
        scenario = get_preset_3x3()
        code_str = export_to_python_code(scenario)
        self.assertIn("NUM_JOBS = 3", code_str)
        self.assertIn("JOBS_DATA = [", code_str)
        self.assertIn("FACTORY_CONFIG = {", code_str)
        
        # Test that exported python code is valid and executable
        namespace = {}
        exec(code_str, namespace)
        self.assertEqual(namespace["NUM_JOBS"], 3)
        self.assertEqual(namespace["NUM_MACHINES"], 3)
        self.assertEqual(len(namespace["JOBS_DATA"]), 3)

        # Test JSON export
        json_str = export_to_json(scenario)
        self.assertIn('"num_jobs": 3', json_str)

    def test_flexible_permutation_no_consecutive_repeats(self):
        # Generate random flexible scenario
        scenario = generate_random_scenario(
            num_jobs=6,
            num_machines=6,
            mode="flexible",
            num_groups=3,
            seed=42
        )
        for job in scenario.jobs:
            group_sequence = [s.group_name for s in job.steps]
            # Ensure every job visits exactly the distinct groups in a permutation
            self.assertEqual(len(group_sequence), 3)
            self.assertEqual(len(set(group_sequence)), 3, f"Duplicate groups found in job route: {group_sequence}")
            # Ensure no consecutive repeats
            for k in range(len(group_sequence) - 1):
                self.assertNotEqual(group_sequence[k], group_sequence[k+1])

    def test_rename_machine_group(self):
        from designer_core import rename_machine_group
        scenario = get_preset_3x3()
        self.assertIn("Work Center A", scenario.machine_groups)
        
        # Add a step that uses Work Center A
        scenario.jobs[0].steps[0].group_name = "Work Center A"
        
        # Rename to "CNC Milling"
        success = rename_machine_group(scenario, "Work Center A", "CNC Milling")
        self.assertTrue(success)
        self.assertNotIn("Work Center A", scenario.machine_groups)
        self.assertIn("CNC Milling", scenario.machine_groups)
        self.assertEqual(scenario.machine_groups["CNC Milling"].name, "CNC Milling")
        
        # Verify job step was automatically updated
        self.assertEqual(scenario.jobs[0].steps[0].group_name, "CNC Milling")

    def test_generate_neural_network_dot(self):
        from designer_core import generate_neural_network_dot
        scenario = get_preset_7x6()
        
        # Test TB (horizontal rows)
        dot_tb = generate_neural_network_dot(scenario, orientation="TB")
        self.assertIn("rankdir=TB;", dot_tb)
        self.assertIn("subgraph cluster_layer_0", dot_tb)
        self.assertIn("rank=same", dot_tb)
        self.assertIn("Input Layer", dot_tb)
        self.assertIn("Output Layer", dot_tb)
        
        # Test LR (vertical columns)
        dot_lr = generate_neural_network_dot(scenario, orientation="LR")
        self.assertIn("rankdir=LR;", dot_lr)
        
        # Test Highlight Job
        dot_hl = generate_neural_network_dot(scenario, highlight_job_id=0, orientation="TB")
        self.assertIn("#EC4899", dot_hl)

if __name__ == "__main__":
    unittest.main()
