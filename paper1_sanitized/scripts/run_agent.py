"""
Run a code agent on SWE-bench instances and collect trajectories.

FIXED: Uses Venus API + OpenAI SDK (no litellm), includes 6.5s rate limiting,
MAX_CALLS budget cap, and incremental save-as-we-go pattern.

Usage:
    python run_agent.py --instance_id django__django-16379 --model gpt-4o-mini
    python run_agent.py --subset data/swebench_subset.json --model gpt-4o-mini --parallel 4
    python run_agent.py --instance_id django__django-16379 --model gpt-4o-mini --scaffold EDIT_A

Supported models:
    - gpt-4o-mini (OpenAI, cheapest)
    - gpt-4.1 (OpenAI, strong)
    - deepseek-v4 (DeepSeek, cross-family)
    - claude-sonnet-4 (Anthropic, optional)
    - gpt-5.5 (OpenAI, ceiling only)
"""

import argparse
import json
import os
import sys
import time
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

# Use OpenAI SDK directly with Venus proxy
from openai import OpenAI, AzureOpenAI


# Venus API configuration
VENUS_PROXY_URL = os.environ.get("VENUS_PROXY_URL", "<REDACTED_URL>")
VENUS_API_KEY = os.environ.get("OPENAI_API_KEY", "<REDACTED_SECRET>")


# Model configurations — all routed through Venus proxy with unified API key
MODEL_CONFIG = {
    "gpt-4o-mini": {
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "model_name": "gpt-4o-mini",
        "max_tokens": 4096,
        "temperature": 0.0,
        "cost_per_1k_input": 0.00015,
        "cost_per_1k_output": 0.0006,
    },
    "gpt-4.1": {
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "model_name": "gpt-4.1",
        "max_tokens": 4096,
        "temperature": 0.0,
        "cost_per_1k_input": 0.002,
        "cost_per_1k_output": 0.008,
    },
    "deepseek-v4": {
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "model_name": "deepseek-v4-pro",
        "max_tokens": 4096,
        "temperature": 0.0,
        "cost_per_1k_input": 0.0005,
        "cost_per_1k_output": 0.002,
    },
    "claude-sonnet-4": {
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "model_name": "claude-sonnet-4-6",
        "max_tokens": 4096,
        "temperature": 0.0,
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
    },
    "gpt-5.5": {
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "model_name": "gpt-5.5",
        "max_tokens": 4096,
        "temperature": 0.0,
        "cost_per_1k_input": 0.005,
        "cost_per_1k_output": 0.015,
    },
}

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
RESULTS_DIR = PROJECT_ROOT / "results"

# Rate limiting: 6.5s per API call
RATE_LIMIT_DELAY_SECONDS = 6.5
# Budget cap: max API calls per batch run
MAX_CALLS_DEFAULT = 1000


def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    # Check scaffolding subdir first
    scaffold_path = PROMPTS_DIR / "scaffolding" / f"{prompt_name}.txt"
    if scaffold_path.exists():
        return scaffold_path.read_text()
    # Check top-level prompts
    prompt_path = PROMPTS_DIR / f"{prompt_name}.txt"
    if prompt_path.exists():
        return prompt_path.read_text()
    raise FileNotFoundError(f"Prompt not found: {prompt_name}")


def load_instance(instance_id: str) -> dict:
    """Load a SWE-bench instance by ID from the full dataset."""
    # Try loading from cached full dataset
    cache_path = DATA_DIR / "swebench_full.json"
    if cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
        for inst in data:
            if inst["instance_id"] == instance_id:
                return inst

    # Load from HuggingFace
    os.environ.setdefault('HF_HOME', '/home/xiankunlin/.cache/huggingface')
    from datasets import load_dataset
    ds = load_dataset('princeton-nlp/SWE-bench_Verified', split='test',
                      cache_dir='/home/xiankunlin/.cache/huggingface/datasets')
    for item in ds:
        if item["instance_id"] == instance_id:
            return dict(item)
    raise ValueError(f"Instance not found: {instance_id}")


def load_subset(subset_path: str) -> list:
    """Load instance IDs from a subset file."""
    with open(subset_path) as f:
        data = json.load(f)
    return [inst["instance_id"] for inst in data["instances"]]


class AgentTrajectory:
    """Collects and stores an agent's trajectory on a task."""

    def __init__(self, instance_id: str, model: str, scaffold: Optional[str] = None):
        self.instance_id = instance_id
        self.model = model
        self.scaffold = scaffold
        self.steps = []
        self.start_time = datetime.now().isoformat()
        self.end_time = None
        self.resolved = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.error = None

    def add_step(self, step: dict):
        """Add a step to the trajectory."""
        self.steps.append(step)

    def finalize(self, resolved: bool, error: Optional[str] = None):
        """Mark trajectory as complete."""
        self.end_time = datetime.now().isoformat()
        self.resolved = resolved
        self.error = error

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "model": self.model,
            "scaffold": self.scaffold,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "resolved": self.resolved,
            "num_steps": len(self.steps),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost,
            "error": self.error,
            "steps": self.steps,
        }

    def save(self, output_dir: Path):
        """Save trajectory to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_scaffold_{self.scaffold}" if self.scaffold else ""
        filename = f"{self.instance_id}{suffix}.json"
        output_path = output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        return output_path


class SWEBenchAgent:
    """
    Agent that interacts with a SWE-bench instance via LLM using Venus API.

    This is a simplified agent that simulates the interaction loop:
    1. Presents the task to the LLM
    2. LLM generates actions (bash commands)
    3. Actions are executed in Docker container
    4. Observations are fed back to the LLM
    5. Repeat until resolved or max steps
    """

    def __init__(self, model: str, max_steps: int = 30, scaffold: Optional[str] = None):
        self.model = model
        self.model_config = MODEL_CONFIG[model]
        self.max_steps = max_steps
        self.scaffold = scaffold
        self.scaffold_prompt = None
        if scaffold:
            self.scaffold_prompt = load_prompt(scaffold)
        
        # Initialize OpenAI client with Venus proxy
        api_key = os.environ.get(self.model_config["api_key_env"], VENUS_API_KEY)
        if not api_key:
            raise ValueError(f"API key not set: {self.model_config['api_key_env']}")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=VENUS_PROXY_URL,
        )

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        return load_prompt("agent_base")

    def _build_initial_message(self, instance: dict) -> str:
        """Build the initial user message with the task."""
        base_prompt = self._build_system_prompt()
        formatted = base_prompt.replace("{problem_statement}", instance["problem_statement"])
        return formatted

    def _call_llm(self, messages: list, trajectory: AgentTrajectory) -> dict:
        """
        Call the LLM via Venus proxy and track token usage.
        
        FIXED: Uses OpenAI SDK with Venus proxy, includes 6.5s rate limiting.
        """
        config = self.model_config
        
        try:
            # Apply rate limiting BEFORE the call
            time.sleep(RATE_LIMIT_DELAY_SECONDS)
            
            # Call LLM via Venus proxy
            response = self.client.chat.completions.create(
                model=config["model_name"],
                messages=messages,
                max_tokens=config["max_tokens"],
                temperature=config["temperature"],
            )

            # Track tokens
            usage = response.usage
            trajectory.total_input_tokens += usage.prompt_tokens
            trajectory.total_output_tokens += usage.completion_tokens
            trajectory.total_cost += (
                usage.prompt_tokens / 1000 * config["cost_per_1k_input"] +
                usage.completion_tokens / 1000 * config["cost_per_1k_output"]
            )

            return {
                "content": response.choices[0].message.content,
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
            }
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}")

    def _parse_action(self, response_text: str) -> dict:
        """
        Parse the LLM's response to extract the action.

        Expected format: The LLM outputs thought + bash command.
        We extract the command to execute.
        """
        # Simple parsing: look for code blocks or command patterns
        lines = response_text.strip().split('\n')
        thought_lines = []
        command_lines = []
        in_code_block = False

        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                command_lines.append(line)
            else:
                thought_lines.append(line)

        thought = '\n'.join(thought_lines).strip()
        command = '\n'.join(command_lines).strip()

        # If no code block found, look for common command patterns
        if not command:
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('$ ') or stripped.startswith('> '):
                    command = stripped[2:]
                    break

        return {
            "thought": thought,
            "command": command if command else "echo 'No command parsed'",
        }

    def _execute_command(self, command: str, instance: dict) -> str:
        """
        Execute a command in the SWE-bench Docker environment.

        NOTE: In the actual experiment, this uses mini-swe-agent's Docker backend.
        For framework testing, we return a placeholder.
        """
        # TODO: Integrate with mini-swe-agent Docker environment
        # For now, this is a placeholder that would be replaced with actual execution
        #
        # In production, this would:
        # 1. Send the command to a Docker container with the repo checked out
        # 2. Execute it via subprocess
        # 3. Return stdout + stderr
        #
        # The actual integration point with mini-swe-agent would be:
        #   from minisweagent.environments.docker import DockerEnvironment
        #   env = DockerEnvironment(image=instance_image)
        #   result = env.execute(command)
        #   return result.output

        return "[PLACEHOLDER: Command would be executed in Docker environment]"

    def _is_submission(self, response_text: str) -> bool:
        """Check if the agent is trying to submit its solution."""
        return "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in response_text

    def run(self, instance: dict) -> AgentTrajectory:
        """Run the agent on a single SWE-bench instance."""
        trajectory = AgentTrajectory(
            instance_id=instance["instance_id"],
            model=self.model,
            scaffold=self.scaffold,
        )

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": self._build_initial_message(instance)},
        ]

        for step_num in range(1, self.max_steps + 1):
            try:
                # Get LLM response
                response = self._call_llm(messages, trajectory)
                response_text = response["content"]

                # Parse action
                action = self._parse_action(response_text)

                # Check for submission
                if self._is_submission(response_text):
                    step_data = {
                        "step": step_num,
                        "thought": action["thought"],
                        "action": "submit",
                        "action_args": "",
                        "observation": "Agent submitted solution",
                        "input_tokens": response["input_tokens"],
                        "output_tokens": response["output_tokens"],
                    }
                    trajectory.add_step(step_data)
                    break

                # Execute command
                observation = self._execute_command(action["command"], instance)

                # Record step
                step_data = {
                    "step": step_num,
                    "thought": action["thought"],
                    "action": "bash",
                    "action_args": action["command"],
                    "observation": observation,
                    "input_tokens": response["input_tokens"],
                    "output_tokens": response["output_tokens"],
                }
                trajectory.add_step(step_data)

                # Add observation to conversation
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": f"Observation:\n{observation}"})

            except Exception as e:
                trajectory.add_step({
                    "step": step_num,
                    "thought": "",
                    "action": "error",
                    "action_args": "",
                    "observation": str(e),
                    "input_tokens": 0,
                    "output_tokens": 0,
                })
                trajectory.finalize(resolved=False, error=str(e))
                return trajectory

        # Evaluate resolution (placeholder — actual evaluation uses SWE-bench harness)
        trajectory.finalize(resolved=None)  # None = needs evaluation
        return trajectory

    def run_with_scaffold(self, instance: dict, failed_trajectory: AgentTrajectory,
                          scaffold_step: Optional[int] = None) -> AgentTrajectory:
        """
        Re-run the agent with scaffolding feedback injected after failure.

        Args:
            instance: The SWE-bench instance
            failed_trajectory: The original failed trajectory
            scaffold_step: Step to inject scaffold at (default: after last step)
        """
        trajectory = AgentTrajectory(
            instance_id=instance["instance_id"],
            model=self.model,
            scaffold=self.scaffold,
        )

        # Build conversation up to failure point
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": self._build_initial_message(instance)},
        ]

        # Replay conversation up to scaffold injection point
        inject_at = scaffold_step or len(failed_trajectory.steps)
        for step in failed_trajectory.steps[:inject_at]:
            # Simulate the original conversation
            messages.append({"role": "assistant", "content": step.get("thought", "")})
            messages.append({"role": "user", "content": f"Observation:\n{step.get('observation', '')}"})
            trajectory.add_step({**step, "replayed": True})

        # Inject scaffolding feedback
        if self.scaffold_prompt:
            messages.append({"role": "user", "content": self.scaffold_prompt})
            trajectory.add_step({
                "step": inject_at + 1,
                "thought": "",
                "action": "scaffold_injection",
                "action_args": self.scaffold,
                "observation": self.scaffold_prompt,
                "input_tokens": 0,
                "output_tokens": 0,
            })

        # Continue with additional steps
        additional_max = min(15, self.max_steps - inject_at)
        for step_num in range(inject_at + 2, inject_at + 2 + additional_max):
            try:
                response = self._call_llm(messages, trajectory)
                response_text = response["content"]
                action = self._parse_action(response_text)

                if self._is_submission(response_text):
                    trajectory.add_step({
                        "step": step_num,
                        "thought": action["thought"],
                        "action": "submit",
                        "action_args": "",
                        "observation": "Agent submitted solution",
                        "input_tokens": response["input_tokens"],
                        "output_tokens": response["output_tokens"],
                    })
                    break

                observation = self._execute_command(action["command"], instance)
                step_data = {
                    "step": step_num,
                    "thought": action["thought"],
                    "action": "bash",
                    "action_args": action["command"],
                    "observation": observation,
                    "input_tokens": response["input_tokens"],
                    "output_tokens": response["output_tokens"],
                }
                trajectory.add_step(step_data)
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": f"Observation:\n{observation}"})

            except Exception as e:
                trajectory.finalize(resolved=False, error=str(e))
                return trajectory

        trajectory.finalize(resolved=None)
        return trajectory


def run_single(instance_id: str, model: str, scaffold: Optional[str] = None,
               output_dir: Optional[str] = None, max_steps: int = 30) -> Path:
    """Run agent on a single instance and save trajectory."""
    instance = load_instance(instance_id)
    agent = SWEBenchAgent(model=model, max_steps=max_steps, scaffold=scaffold)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running {model} on {instance_id}", end="")
    if scaffold:
        print(f" with scaffold={scaffold}", end="")
    print("...")

    trajectory = agent.run(instance)

    # Save incrementally
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = DATA_DIR / "trajectories" / model
    saved = trajectory.save(out_path)
    print(f"  -> Saved to {saved} ({trajectory.total_cost:.4f}$, {len(trajectory.steps)} steps)")
    return saved


def run_batch(subset_path: str, model: str, scaffold: Optional[str] = None,
              output_dir: Optional[str] = None, max_steps: int = 30,
              parallel: int = 1, max_retries: int = 3, max_calls: int = MAX_CALLS_DEFAULT):
    """
    Run agent on all instances in a subset file.
    
    FIXED: Includes MAX_CALLS budget cap to prevent runaway costs.
    """
    instance_ids = load_subset(subset_path)
    print(f"Running {model} on {len(instance_ids)} instances (parallel={parallel}, max_calls={max_calls})")

    results = []
    calls_made = 0
    
    for i, instance_id in enumerate(instance_ids):
        # Budget cap: stop if we exceed MAX_CALLS
        if calls_made >= max_calls:
            print(f"\n[BUDGET CAP] Reached max_calls limit ({max_calls}). Stopping.")
            results.append({
                "instance_id": instance_id,
                "status": "skipped",
                "reason": f"Budget cap reached after {calls_made} calls"
            })
            continue
        
        print(f"[{i+1}/{len(instance_ids)}] (calls: {calls_made}/{max_calls}) ", end="")
        for attempt in range(max_retries):
            try:
                path = run_single(instance_id, model, scaffold, output_dir, max_steps)
                results.append({"instance_id": instance_id, "status": "success", "path": str(path)})
                calls_made += max_steps  # Rough estimate: max_steps calls per instance
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"  Retry {attempt+1}/{max_retries} after {wait}s: {e}")
                    time.sleep(wait)
                else:
                    print(f"  FAILED after {max_retries} attempts: {e}")
                    results.append({"instance_id": instance_id, "status": "failed", "error": str(e)})

    # Save batch results summary incrementally
    summary_path = RESULTS_DIR / f"batch_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump({
            "model": model,
            "scaffold": scaffold,
            "subset": subset_path,
            "total": len(instance_ids),
            "success": sum(1 for r in results if r["status"] == "success"),
            "failed": sum(1 for r in results if r["status"] == "failed"),
            "skipped": sum(1 for r in results if r["status"] == "skipped"),
            "total_calls_made": calls_made,
            "max_calls_limit": max_calls,
            "results": results,
        }, f, indent=2)
    print(f"\nBatch complete. Summary: {summary_path}")
    print(f"Total calls made: {calls_made}/{max_calls}")
    return summary_path


def main():
    parser = argparse.ArgumentParser(description="Run SWE-bench agent and collect trajectories")
    parser.add_argument("--instance_id", "-i", type=str, help="Single instance ID to run")
    parser.add_argument("--subset", "-s", type=str, help="Path to subset JSON file")
    parser.add_argument("--model", "-m", type=str, required=True,
                        choices=list(MODEL_CONFIG.keys()),
                        help="Model to use")
    parser.add_argument("--scaffold", type=str, default=None,
                        help="Scaffolding prompt name (e.g., LOC_A_broaden_search)")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                        help="Output directory for trajectories")
    parser.add_argument("--max_steps", type=int, default=30,
                        help="Maximum steps per run")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel runs (for batch mode)")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Max retries on API failure")
    parser.add_argument("--max_calls", type=int, default=MAX_CALLS_DEFAULT,
                        help=f"Max API calls budget (default: {MAX_CALLS_DEFAULT})")

    args = parser.parse_args()

    if not args.instance_id and not args.subset:
        parser.error("Must specify either --instance_id or --subset")

    if args.instance_id:
        run_single(args.instance_id, args.model, args.scaffold, args.output_dir, args.max_steps)
    else:
        run_batch(args.subset, args.model, args.scaffold, args.output_dir,
                  args.max_steps, args.parallel, args.max_retries, args.max_calls)


if __name__ == "__main__":
    main()
