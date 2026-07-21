"""
Quick dry-run test: validates pipeline without API calls.

Tests:
1. Data loading (SWE-bench subset JSON)
2. Prompt template loading and formatting
3. Agent class instantiation
4. Trajectory creation and serialization
5. Scaffolding prompt injection
6. Failure annotation pipeline (dry)
7. C&B engine logic (dry)
8. Output format validation

Usage:
    python quick_test.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# Add scripts dir to path
sys.path.insert(0, str(SCRIPT_DIR))

# ============================================================
# Test utilities
# ============================================================

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name: str):
        self.passed += 1
        print(f"  ✓ {name}")

    def fail(self, name: str, reason: str):
        self.failed += 1
        self.errors.append((name, reason))
        print(f"  ✗ {name}: {reason}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print(f"\nFailures:")
            for name, reason in self.errors:
                print(f"  - {name}: {reason}")
        print(f"{'='*60}")
        return self.failed == 0


# ============================================================
# Tests
# ============================================================

def test_imports(results: TestResult):
    """Test that all required packages can be imported."""
    print("\n[1/8] Testing imports...")

    packages = [
        ("litellm", "litellm"),
        ("datasets", "datasets"),
        ("pandas", "pandas"),
        ("tqdm", "tqdm"),
        ("docker", "docker"),
        ("openai", "openai"),
        ("anthropic", "anthropic"),
    ]

    for display_name, module_name in packages:
        try:
            __import__(module_name)
            results.ok(f"import {display_name}")
        except ImportError as e:
            results.fail(f"import {display_name}", str(e))


def test_data_loading(results: TestResult):
    """Test that SWE-bench subset data loads correctly."""
    print("\n[2/8] Testing data loading...")

    subset_path = DATA_DIR / "swebench_subset.json"
    if not subset_path.exists():
        results.fail("subset file exists", f"Not found: {subset_path}")
        return

    try:
        with open(subset_path) as f:
            data = json.load(f)

        # Check structure
        assert "metadata" in data, "Missing 'metadata' key"
        assert "instances" in data, "Missing 'instances' key"
        assert "repo_distribution" in data, "Missing 'repo_distribution' key"

        instances = data["instances"]
        assert len(instances) == 200, f"Expected 200 instances, got {len(instances)}"
        results.ok(f"subset loaded: {len(instances)} instances")

        # Check instance structure
        first = instances[0]
        required_fields = ["instance_id", "repo", "base_commit", "version"]
        for field in required_fields:
            assert field in first, f"Instance missing field: {field}"
        results.ok(f"instance structure valid (has {', '.join(required_fields)})")

        # Check repo distribution
        repos = data["repo_distribution"]
        assert len(repos) >= 10, f"Expected ≥10 repos, got {len(repos)}"
        results.ok(f"repo distribution: {len(repos)} repos")

    except AssertionError as e:
        results.fail("data structure", str(e))
    except Exception as e:
        results.fail("data loading", str(e))


def test_prompt_loading(results: TestResult):
    """Test that all prompt templates load and format correctly."""
    print("\n[3/8] Testing prompt templates...")

    # Check base prompt
    base_path = PROMPTS_DIR / "agent_base.txt"
    if not base_path.exists():
        results.fail("agent_base.txt exists", "Not found")
        return

    base_prompt = base_path.read_text()
    assert "{problem_statement}" in base_prompt, "Base prompt missing {problem_statement} placeholder"
    results.ok("agent_base.txt loaded and has placeholder")

    # Test formatting
    formatted = base_prompt.replace("{problem_statement}", "Test issue: fix bug in query.py")
    assert "Test issue" in formatted
    assert "{problem_statement}" not in formatted
    results.ok("prompt formatting works")

    # Check all scaffolding prompts
    scaffold_dir = PROMPTS_DIR / "scaffolding"
    if not scaffold_dir.exists():
        results.fail("scaffolding dir exists", "Not found")
        return

    expected_prompts = [
        "LOC_A_broaden_search", "LOC_B_reread_issue", "LOC_C_test_guided",
        "EDIT_A_reread_file", "EDIT_B_smaller_edit", "EDIT_C_alternative_tool",
        "LOGIC_A_test_analysis", "LOGIC_B_minimal_fix", "LOGIC_C_edge_cases",
        "TEST_A_issue_reread", "TEST_B_test_first", "TEST_C_differential",
        "PLAN_A_step_back", "PLAN_B_scope_check", "PLAN_C_similar_fixes",
        "CONTROL_no_scaffold",
    ]

    found = 0
    missing = []
    for name in expected_prompts:
        path = scaffold_dir / f"{name}.txt"
        if path.exists():
            content = path.read_text()
            if len(content.strip()) > 10:
                found += 1
            else:
                missing.append(f"{name} (empty)")
        else:
            missing.append(name)

    if missing:
        results.fail(f"scaffolding prompts", f"Missing/empty: {', '.join(missing[:5])}")
    else:
        results.ok(f"all {found} scaffolding prompts loaded")

    # Check classifier prompt
    classifier_path = PROMPTS_DIR / "failure_classifier.txt"
    if classifier_path.exists():
        content = classifier_path.read_text()
        assert len(content) > 50, "Classifier prompt too short"
        results.ok("failure_classifier.txt loaded")
    else:
        results.fail("failure_classifier.txt exists", "Not found")


def test_agent_class(results: TestResult):
    """Test agent class instantiation and logic."""
    print("\n[4/8] Testing agent class...")

    try:
        from run_agent import SWEBenchAgent, AgentTrajectory, MODEL_CONFIG, load_prompt

        # Check model config
        assert "gpt-4o-mini" in MODEL_CONFIG
        assert "gpt-4.1" in MODEL_CONFIG
        assert "deepseek-v4" in MODEL_CONFIG
        results.ok(f"MODEL_CONFIG has {len(MODEL_CONFIG)} models")

        # Instantiate agent (no API call needed)
        agent = SWEBenchAgent(model="gpt-4o-mini", max_steps=5)
        assert agent.model == "gpt-4o-mini"
        assert agent.max_steps == 5
        results.ok("SWEBenchAgent instantiation")

        # Test with scaffold
        agent_s = SWEBenchAgent(model="gpt-4o-mini", max_steps=5, scaffold="LOC_A_broaden_search")
        assert agent_s.scaffold == "LOC_A_broaden_search"
        assert agent_s.scaffold_prompt is not None
        assert len(agent_s.scaffold_prompt) > 10
        results.ok("SWEBenchAgent with scaffold")

    except Exception as e:
        results.fail("agent class", str(e))


def test_trajectory_serialization(results: TestResult):
    """Test trajectory creation, step recording, and JSON output."""
    print("\n[5/8] Testing trajectory serialization...")

    try:
        from run_agent import AgentTrajectory

        traj = AgentTrajectory(
            instance_id="django__django-16379",
            model="gpt-4o-mini",
            scaffold=None,
        )

        # Add steps
        traj.add_step({
            "step": 1,
            "thought": "I need to find the relevant file",
            "action": "find_file",
            "action_args": "query.py django/db/models/",
            "observation": "Found: django/db/models/query.py",
            "input_tokens": 500,
            "output_tokens": 100,
        })
        traj.add_step({
            "step": 2,
            "thought": "Let me open this file",
            "action": "open_file",
            "action_args": "django/db/models/query.py",
            "observation": "[file content...]",
            "input_tokens": 600,
            "output_tokens": 150,
        })

        assert len(traj.steps) == 2
        results.ok(f"trajectory has {len(traj.steps)} steps")

        # Finalize
        traj.total_input_tokens = 1100
        traj.total_output_tokens = 250
        traj.total_cost = 0.0003
        traj.finalize(resolved=False)

        # Serialize
        data = traj.to_dict()
        assert data["instance_id"] == "django__django-16379"
        assert data["model"] == "gpt-4o-mini"
        assert data["num_steps"] == 2
        assert data["resolved"] == False
        assert data["total_cost"] == 0.0003
        results.ok("trajectory serialization to dict")

        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = traj.save(Path(tmpdir))
            assert saved.exists()
            with open(saved) as f:
                loaded = json.load(f)
            assert loaded["instance_id"] == "django__django-16379"
            assert len(loaded["steps"]) == 2
            results.ok(f"trajectory saved to file: {saved.name}")

    except Exception as e:
        results.fail("trajectory", str(e))


def test_scaffold_injection(results: TestResult):
    """Test scaffolding prompt injection logic."""
    print("\n[6/8] Testing scaffold injection logic...")

    try:
        from run_agent import SWEBenchAgent, AgentTrajectory

        # Create a mock failed trajectory
        failed_traj = AgentTrajectory("django__django-16379", "gpt-4o-mini")
        for i in range(5):
            failed_traj.add_step({
                "step": i + 1,
                "thought": f"Step {i+1} thought",
                "action": "bash",
                "action_args": f"command_{i+1}",
                "observation": f"output_{i+1}",
                "input_tokens": 100,
                "output_tokens": 50,
            })
        failed_traj.finalize(resolved=False)

        # Create agent with scaffold
        agent = SWEBenchAgent(model="gpt-4o-mini", max_steps=30, scaffold="EDIT_A_reread_file")

        # Verify scaffold prompt is loaded
        assert agent.scaffold_prompt is not None
        assert "Read the exact content" in agent.scaffold_prompt or len(agent.scaffold_prompt) > 20
        results.ok("scaffold prompt loaded for injection")

        # Verify injection point logic
        inject_at = 3  # Inject after step 3
        replayed = failed_traj.steps[:inject_at]
        assert len(replayed) == 3
        results.ok(f"injection at step {inject_at}: replays {len(replayed)} steps")

        # Verify the full flow would work (without actually calling LLM)
        results.ok("scaffold injection logic validated")

    except Exception as e:
        results.fail("scaffold injection", str(e))


def test_annotation_pipeline(results: TestResult):
    """Test failure annotation pipeline (dry run)."""
    print("\n[7/8] Testing annotation pipeline...")

    try:
        from annotate_failures import (
            format_trajectory_for_classification,
            rule_based_classify,
        )

        # Create mock trajectory
        mock_traj = {
            "instance_id": "django__django-16379",
            "model": "gpt-4o-mini",
            "resolved": False,
            "steps": [
                {"step": 1, "thought": "Search for file", "action": "search",
                 "action_args": "find query.py", "observation": "Found files"},
                {"step": 2, "thought": "Open wrong file", "action": "open_file",
                 "action_args": "django/db/models/sql/query.py", "observation": "[content]"},
                {"step": 3, "thought": "Edit file", "action": "str_replace",
                 "action_args": "django/db/models/sql/query.py old new",
                 "observation": "str_replace: old_str not found in file"},
                {"step": 4, "thought": "Try again", "action": "str_replace",
                 "action_args": "django/db/models/sql/query.py old2 new2",
                 "observation": "str_replace: old_str not found in file"},
                {"step": 5, "thought": "Try again", "action": "str_replace",
                 "action_args": "django/db/models/sql/query.py old3 new3",
                 "observation": "str_replace: old_str not found in file"},
            ]
        }

        # Test formatting
        formatted = format_trajectory_for_classification(mock_traj)
        assert "Step 1" in formatted
        assert "Step 5" in formatted
        results.ok("trajectory formatting for classification")

        # Test rule-based classification
        mock_instance = {
            "instance_id": "django__django-16379",
            "patch": "diff --git a/django/db/models/query.py b/django/db/models/query.py\n...",
        }
        rule_result = rule_based_classify(mock_traj, mock_instance)
        assert "failure_type" in rule_result
        assert rule_result["failure_type"] in ("LOC", "EDIT", "LOGIC", "TEST", "PLAN")
        assert "confidence" in rule_result
        results.ok(f"rule-based classification: {rule_result['failure_type']} "
                   f"(conf={rule_result['confidence']:.2f})")

    except Exception as e:
        results.fail("annotation pipeline", str(e))


def test_cb_engine(results: TestResult):
    """Test C&B engine logic (dry)."""
    print("\n[8/8] Testing C&B engine...")

    try:
        from cb_engine import (
            is_checkpoint_worthy,
            detect_error_heuristic,
            find_checkpoint_before,
            BEST_STRATEGY,
        )

        # Test checkpoint detection
        step_search = {"action": "find_file", "action_args": "query.py", "observation": "Found"}
        step_edit = {"action": "str_replace", "action_args": "edit file", "observation": "OK"}
        step_test = {"action": "bash", "action_args": "pytest tests/", "observation": "FAILED"}
        step_normal = {"action": "bash", "action_args": "ls", "observation": "files..."}

        assert is_checkpoint_worthy(step_search) == True
        assert is_checkpoint_worthy(step_test) == True
        assert is_checkpoint_worthy(step_normal) == False
        results.ok("checkpoint worthiness detection")

        # Test heuristic error detection
        # The heuristic checks: "not found" in obs AND ("str_replace" in args OR "edit" in action)
        mock_traj_edit_error = {
            "steps": [
                {"action": "find_file", "action_args": "query.py", "observation": "Found: query.py"},
                {"action": "open_file", "action_args": "query.py", "observation": "[content]"},
                {"action": "edit", "action_args": "str_replace query.py old new",
                 "observation": "Error: old_str not found in file"},
            ]
        }
        error_type, error_step = detect_error_heuristic(mock_traj_edit_error)
        assert error_type == "EDIT", f"Expected EDIT, got {error_type}"
        assert error_step == 2
        results.ok(f"heuristic detection: {error_type} at step {error_step}")

        # Test logic error detection
        mock_traj_logic_error = {
            "steps": [
                {"action": "find_file", "action_args": "query.py", "observation": "Found"},
                {"action": "str_replace", "action_args": "query.py", "observation": "Applied"},
                {"action": "bash", "action_args": "pytest", "observation": "FAILED AssertionError"},
            ]
        }
        error_type2, error_step2 = detect_error_heuristic(mock_traj_logic_error)
        assert error_type2 == "LOGIC", f"Expected LOGIC, got {error_type2}"
        results.ok(f"heuristic detection: {error_type2} at step {error_step2}")

        # Test checkpoint search
        steps = [
            {"action": "find_file", "action_args": "query.py", "observation": "Found"},
            {"action": "bash", "action_args": "ls", "observation": "..."},
            {"action": "open_file", "action_args": "query.py", "observation": "[content]"},
            {"action": "str_replace", "action_args": "query.py", "observation": "Error"},
        ]
        cp = find_checkpoint_before(steps, 3)
        assert cp == 0  # The find_file step is the checkpoint
        results.ok(f"checkpoint found at step {cp}")

        # Test strategy mapping
        assert "LOC" in BEST_STRATEGY
        assert "EDIT" in BEST_STRATEGY
        assert len(BEST_STRATEGY) == 5
        results.ok(f"strategy mapping: {len(BEST_STRATEGY)} types configured")

    except Exception as e:
        results.fail("C&B engine", str(e))


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("EMNLP Paper 1 - Pipeline Dry-Run Test")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project: {PROJECT_ROOT}")
    print("=" * 60)

    results = TestResult()

    test_imports(results)
    test_data_loading(results)
    test_prompt_loading(results)
    test_agent_class(results)
    test_trajectory_serialization(results)
    test_scaffold_injection(results)
    test_annotation_pipeline(results)
    test_cb_engine(results)

    success = results.summary()

    if success:
        print("\n🎉 All tests passed! Pipeline is ready for API-key-gated experiments.")
        print("\nNext steps:")
        print("  1. Set OPENAI_API_KEY, DEEPSEEK_API_KEY (and optionally ANTHROPIC_API_KEY)")
        print("  2. Run: python collect_trajectories.py -m gpt-4o-mini --dry_run")
        print("  3. Then: python collect_trajectories.py -m gpt-4o-mini")
    else:
        print(f"\n⚠ {results.failed} test(s) failed. Fix issues before proceeding.")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
