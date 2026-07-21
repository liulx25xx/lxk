"""
Annotate failure types for collected trajectories using LLM-as-classifier.

FIXED: Uses Venus API + OpenAI SDK (no litellm), includes 6.5s rate limiting,
MAX_CALLS budget cap, and incremental result saving.

Takes failed trajectories and classifies them into 5 failure types:
LOC, EDIT, LOGIC, TEST, PLAN

Usage:
    python annotate_failures.py --model gpt-4o-mini --trajectories ../data/trajectories/gpt-4o-mini/
    python annotate_failures.py --all_models  # annotate all models
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from run_agent import MODEL_CONFIG

from openai import OpenAI


# Venus API configuration
VENUS_PROXY_URL = os.environ.get("VENUS_PROXY_URL", "<REDACTED_URL>")
# Rate limiting: 6.5s per API call
RATE_LIMIT_DELAY_SECONDS = 6.5
# Budget cap: max classification API calls
MAX_CALLS_DEFAULT = 500

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = PROJECT_ROOT / "prompts"


def load_classifier_prompt() -> str:
    """Load the failure classifier prompt template."""
    prompt_path = PROMPTS_DIR / "failure_classifier.txt"
    return prompt_path.read_text()


def load_trajectory(traj_path: Path) -> dict:
    """Load a trajectory file."""
    with open(traj_path) as f:
        return json.load(f)


def format_trajectory_for_classification(trajectory: dict, max_steps: int = 20) -> str:
    """Format a trajectory for the classifier prompt."""
    steps = trajectory.get("steps", [])
    # Take last N steps if too long (recent context is most informative)
    if len(steps) > max_steps:
        steps = steps[-max_steps:]

    formatted = []
    for step in steps:
        formatted.append(
            f"Step {step.get('step', '?')}:\n"
            f"  Thought: {step.get('thought', '')[:200]}\n"
            f"  Action: {step.get('action', '')} {step.get('action_args', '')[:100]}\n"
            f"  Observation: {step.get('observation', '')[:300]}"
        )
    return '\n\n'.join(formatted)


def classify_failure(trajectory: dict, instance: dict, client: OpenAI,
                     classifier_model: str = "gpt-4o-mini") -> dict:
    """
    Classify a failed trajectory using LLM via Venus proxy.

    Args:
        trajectory: The failed trajectory dict
        instance: The SWE-bench instance (with problem_statement, patch, etc.)
        client: OpenAI client configured with Venus proxy
        classifier_model: Model to use for classification

    Returns:
        Classification result dict
    """
    prompt_template = load_classifier_prompt()

    # Extract gold patch files
    patch = instance.get("patch", "")
    gold_files = []
    for line in patch.split('\n'):
        if line.startswith('diff --git'):
            parts = line.split()
            if len(parts) >= 4:
                gold_files.append(parts[3].lstrip('b/'))

    # Format the prompt
    traj_text = format_trajectory_for_classification(trajectory)
    prompt = prompt_template.format(
        issue_text=instance.get("problem_statement", "")[:2000],
        gold_patch_files=', '.join(gold_files),
        trajectory=traj_text,
        test_output=trajectory.get("steps", [{}])[-1].get("observation", "")[:500] if trajectory.get("steps") else "",
    )

    # Call LLM via Venus proxy
    try:
        # Apply rate limiting BEFORE the call
        time.sleep(RATE_LIMIT_DELAY_SECONDS)
        
        response = client.chat.completions.create(
            model=classifier_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.0,
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON from response
        # Handle markdown code blocks
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()

        result = json.loads(content)
        result["raw_response"] = response.choices[0].message.content
        result["tokens_used"] = {
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
        }
        return result

    except json.JSONDecodeError as e:
        return {
            "failure_type": "UNKNOWN",
            "confidence": 0.0,
            "evidence": f"Failed to parse LLM response: {e}",
            "raw_response": content if 'content' in dir() else "",
        }
    except Exception as e:
        return {
            "failure_type": "ERROR",
            "confidence": 0.0,
            "evidence": f"Classification error: {e}",
        }


def rule_based_classify(trajectory: dict, instance: dict) -> dict:
    """
    Rule-based classification as cross-check.

    Simple heuristic rules that can confirm/conflict with LLM classification.
    """
    steps = trajectory.get("steps", [])
    patch = instance.get("patch", "")

    # Extract gold patch files
    gold_files = set()
    for line in patch.split('\n'):
        if line.startswith('diff --git'):
            parts = line.split()
            if len(parts) >= 4:
                gold_files.add(parts[3].lstrip('b/'))

    # Check various signals
    edit_errors = 0
    files_touched = set()
    search_actions = 0
    repeated_actions = 0

    for i, step in enumerate(steps):
        obs = step.get("observation", "")
        action = step.get("action", "")
        args = step.get("action_args", "")

        # Edit errors
        if "not found" in obs.lower() or "no match" in obs.lower():
            edit_errors += 1

        # Track files edited
        if action in ("str_replace", "edit", "bash") and "/" in args:
            # Simple extraction of file paths
            for word in args.split():
                if "/" in word and "." in word:
                    files_touched.add(word.strip("'\""))

        # Repeated actions (stuck)
        if i >= 2:
            if (steps[i].get("action") == steps[i-1].get("action") ==
                steps[i-2].get("action")):
                repeated_actions += 1

    # Classification logic
    if edit_errors >= 3:
        return {"failure_type": "EDIT", "confidence": 0.7, "evidence": f"{edit_errors} edit errors"}

    # Check if agent ever touched gold files
    touched_gold = bool(files_touched & gold_files)
    if not touched_gold and gold_files:
        return {"failure_type": "LOC", "confidence": 0.6, "evidence": "Never touched gold patch files"}

    if repeated_actions >= 2:
        return {"failure_type": "PLAN", "confidence": 0.5, "evidence": "Repeated actions (stuck loop)"}

    # Default: LOGIC (most common when location is right)
    return {"failure_type": "LOGIC", "confidence": 0.4, "evidence": "Default (no strong signal)"}


def annotate_model_trajectories(model: str, traj_dir: Path, output_path: Path,
                                dry_run: bool = False, max_calls: int = MAX_CALLS_DEFAULT):
    """
    Annotate all failed trajectories for a model.
    
    FIXED: Includes incremental saving and MAX_CALLS budget cap.
    """
    # Load instances for gold patch info
    os.environ.setdefault('HF_HOME', '/home/xiankunlin/.cache/huggingface')
    from datasets import load_dataset
    ds = load_dataset('princeton-nlp/SWE-bench_Verified', split='test',
                      cache_dir='/home/xiankunlin/.cache/huggingface/datasets')
    instances = {item["instance_id"]: dict(item) for item in ds}

    # Find failed trajectories
    traj_files = list(traj_dir.glob("*.json"))
    print(f"Found {len(traj_files)} trajectory files in {traj_dir}")

    # Initialize OpenAI client with Venus proxy
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not dry_run:
        raise ValueError("OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=api_key, base_url=VENUS_PROXY_URL) if not dry_run else None

    annotations = []
    total_cost = 0.0
    calls_made = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # File for incremental saves during annotation
    partial_output = output_path.with_suffix('.partial.json')

    for i, traj_file in enumerate(traj_files):
        # Budget cap check
        if calls_made >= max_calls:
            print(f"\n[BUDGET CAP] Reached max_calls limit ({max_calls}). Stopping.")
            break
        
        traj = load_trajectory(traj_file)

        # Skip successful ones
        if traj.get("resolved") is True:
            continue

        instance_id = traj.get("instance_id", traj_file.stem)
        instance = instances.get(instance_id)
        if not instance:
            print(f"  Warning: instance {instance_id} not found in dataset")
            continue

        print(f"  [{i+1}/{len(traj_files)}] (calls: {calls_made}/{max_calls}) {instance_id}...", end="")

        if dry_run:
            print(" [DRY RUN]")
            continue

        # LLM classification
        llm_result = classify_failure(traj, instance, client)
        calls_made += 1

        # Rule-based classification
        rule_result = rule_based_classify(traj, instance)

        # Merge
        annotation = {
            "instance_id": instance_id,
            "model": model,
            "llm_classification": llm_result,
            "rule_classification": rule_result,
            # Final label: use LLM by default, rule overrides for LOC/EDIT if high confidence
            "failure_type": llm_result.get("failure_type", "UNKNOWN"),
            "confidence": llm_result.get("confidence", 0.0),
            "first_error_step": llm_result.get("first_error_step"),
            "recommended_strategy": llm_result.get("recommended_strategy"),
        }

        # Rule override logic
        if (rule_result["failure_type"] in ("LOC", "EDIT") and
            rule_result["confidence"] >= 0.7 and
            llm_result.get("failure_type") != rule_result["failure_type"]):
            annotation["failure_type"] = rule_result["failure_type"]
            annotation["override_reason"] = "rule_high_confidence"

        annotations.append(annotation)
        tokens = llm_result.get("tokens_used", {})
        cost = (tokens.get("input", 0) / 1000 * 0.00015 +
                tokens.get("output", 0) / 1000 * 0.0006)
        total_cost += cost
        print(f" -> {annotation['failure_type']} ({annotation['confidence']:.2f})")
        
        # Incremental save every 10 annotations
        if (i + 1) % 10 == 0 or i == len(traj_files) - 1:
            with open(partial_output, 'w') as f:
                json.dump(annotations, f, indent=2)
            print(f"    [Incremental save: {len(annotations)} annotations]")

    # Final save
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=2)

    # Print distribution
    type_dist = Counter(a["failure_type"] for a in annotations)
    print(f"\nAnnotation complete:")
    print(f"  Total annotated: {len(annotations)}")
    print(f"  Total calls made: {calls_made}/{max_calls}")
    print(f"  Total cost: ${total_cost:.4f}")
    print(f"  Distribution:")
    for ftype, count in type_dist.most_common():
        print(f"    {ftype}: {count} ({count/len(annotations)*100:.1f}%)" if annotations else f"    {ftype}: {count}")
    print(f"  Saved to: {output_path}")
    
    # Clean up partial
    if partial_output.exists():
        partial_output.unlink()


def main():
    parser = argparse.ArgumentParser(description="Annotate failure types")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o-mini")
    parser.add_argument("--trajectories", "-t", type=str, default=None,
                        help="Path to trajectory directory")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--all_models", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_calls", type=int, default=MAX_CALLS_DEFAULT,
                        help=f"Max classification calls (default: {MAX_CALLS_DEFAULT})")

    args = parser.parse_args()

    if args.all_models:
        models = ["gpt-4o-mini", "gpt-4.1", "deepseek-v4"]
    else:
        models = [args.model]

    for model in models:
        traj_dir = Path(args.trajectories) if args.trajectories else DATA_DIR / "trajectories" / model
        output_path = Path(args.output) if args.output else DATA_DIR / "annotations" / f"failure_types_{model}.json"

        if traj_dir.exists():
            print(f"\n{'='*60}")
            print(f"Annotating: {model}")
            print(f"{'='*60}")
            annotate_model_trajectories(model, traj_dir, output_path, args.dry_run, args.max_calls)
        else:
            print(f"Skipping {model}: no trajectories at {traj_dir}")


if __name__ == "__main__":
    main()
