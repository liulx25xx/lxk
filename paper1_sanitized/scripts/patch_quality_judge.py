#!/usr/bin/env python3
"""
Patch-quality judge: GPT-4o evaluates whether each response would likely lead to a correct fix.
Binary yes/no — closer to E2E than 0-3 rubric.
"""
import json, re, time, sys
from pathlib import Path
from collections import defaultdict
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent
client = OpenAI(api_key="<REDACTED_SECRET>", base_url="<REDACTED_URL>")
DELAY = 6.5  # slightly more conservative for gpt-4o

def judge_one(instance_id, failure_type, gold_files, response):
    prompt = f"""You are an expert software engineer evaluating a recovery action for a failed code agent.

Issue: {instance_id}
Failure type: {failure_type}
Files that need fixing: {', '.join(gold_files)}

The agent's next action after receiving recovery guidance:
{response[:500]}

Question: Would this action likely lead the agent toward a CORRECT fix of the bug?
Consider: Does it target the right file? Is it a productive step? Would continuing from here plausibly resolve the issue?

Answer ONLY with a JSON object: {{"likely_correct": true/false, "reason": "one sentence"}}"""
    
    time.sleep(DELAY)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role":"user","content":prompt}],
            max_tokens=100, temperature=0)
        text = resp.choices[0].message.content or ""
        match = re.search(r'\{[^}]+\}', text)
        if match:
            return json.loads(match.group())
    except:
        time.sleep(8)
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role":"user","content":prompt}],
                max_tokens=100, temperature=0)
            text = resp.choices[0].message.content or ""
            match = re.search(r'\{[^}]+\}', text)
            if match:
                return json.loads(match.group())
        except:
            pass
    return None

def main():
    phase3 = json.load(open(PROJECT_ROOT / "results/phase3_full_scaffold/full_results.json"))
    annots = json.load(open(PROJECT_ROOT / "results/phase0_annotations/phase0_v2_annotations.json"))
    gold_map = {a['instance_id']: a['gold_files'] for a in annots['annotations']}
    
    out_path = PROJECT_ROOT / "results/patch_quality_judge.json"
    if out_path.exists():
        existing = json.load(open(out_path))
        done_keys = set(f"{r['instance_id']}_{r['strategy']}" for r in existing['results'])
        results = existing['results']
    else:
        done_keys = set()
        results = []
    
    batch_done = 0
    MAX_BATCH = 10
    
    for r in phase3['results']:
        key = f"{r['instance_id']}_{r['strategy']}"
        if key in done_keys:
            continue
        if batch_done >= MAX_BATCH:
            break
        if r['instance_id'] not in gold_map:
            continue
        
        gold_files = gold_map[r['instance_id']]
        judgment = judge_one(r['instance_id'], r['failure_type'], gold_files, r['response'])
        
        if judgment:
            results.append({
                "instance_id": r['instance_id'],
                "failure_type": r['failure_type'],
                "strategy": r['strategy'],
                "score": r['eval']['score'],
                "likely_correct": judgment.get('likely_correct', False),
                "reason": judgment.get('reason', '')
            })
            done_keys.add(key)
            batch_done += 1
    
    with open(out_path, 'w') as f:
        json.dump({"n": len(results), "results": results}, f, indent=2)
    print(f"Saved: {len(results)} total ({batch_done} new)", flush=True)

if __name__ == "__main__":
    main()
