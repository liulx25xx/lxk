#!/usr/bin/env python3
"""Extend blinded judge in small batches (10 per run) to avoid timeout."""
import json, re, time, random, sys, numpy as np
from pathlib import Path
from openai import OpenAI
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).parent.parent
client = OpenAI(api_key="<REDACTED_SECRET>", base_url="<REDACTED_URL>")
BATCH_SIZE = 10

def judge_one(instance_id, gold_files, response):
    prompt = (f"Evaluate this code agent recovery response.\n"
              f"Issue: {instance_id}\n"
              f"Correct files: {', '.join(gold_files)}\n"
              f"Response: {response}\n\n"
              f"Rate (1=yes, 0=no):\n"
              f"1. FILE_RELEVANT: References correct files?\n"
              f"2. ACTIONABLE: Contains executable action?\n"
              f"3. PROGRESS: Moves toward fix?\n\n"
              f'Output ONLY JSON: {{"file_relevant":X,"actionable":X,"progress":X,"total":X}}')
    for attempt in range(3):
        time.sleep(6.5)
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role":"user","content":prompt}],
                max_tokens=80, temperature=0)
            text = resp.choices[0].message.content or ""
            match = re.search(r'\{[^}]+\}', text)
            if match:
                return json.loads(match.group())
        except Exception as e:
            print(f"  Err: {e}", file=sys.stderr, flush=True)
            time.sleep(10)
    return None

def main():
    expanded_path = PROJECT_ROOT / "results/llm_judge_expanded.json"
    existing = json.load(open(expanded_path))
    existing_keys = set(f"{r['instance_id']}_{r['strategy']}" for r in existing['results'])
    
    phase3 = json.load(open(PROJECT_ROOT / "results/phase3_full_scaffold/full_results.json"))
    annots = json.load(open(PROJECT_ROOT / "results/phase0_annotations/phase0_v2_annotations.json"))
    gold_map = {a['instance_id']: a['gold_files'] for a in annots['annotations']}
    
    remaining = [r for r in phase3['results']
                 if f"{r['instance_id']}_{r['strategy']}" not in existing_keys
                 and r['instance_id'] in gold_map]
    
    random.seed(456)
    random.shuffle(remaining)
    batch = remaining[:BATCH_SIZE]
    
    print(f"Batch: {len(batch)} calls (have {existing['n_judged']} already)", flush=True)
    
    new_results = []
    for i, r in enumerate(batch):
        result = judge_one(r['instance_id'], gold_map[r['instance_id']], r['response'][:300])
        if result:
            new_results.append({
                "instance_id": r['instance_id'],
                "failure_type": r['failure_type'],
                "strategy": r['strategy'],
                "regex_eval": r['eval'],
                "regex_score": r['eval']['score'],
                "llm_eval": result,
                "llm_score": result.get('total', 0)
            })
            print(f"  {i+1}/{len(batch)} ok", flush=True)
        else:
            print(f"  {i+1}/{len(batch)} FAILED", flush=True)
    
    # Save
    all_results = existing['results'] + new_results
    regex_scores = [r['regex_score'] for r in all_results]
    llm_scores = [r['llm_score'] for r in all_results]
    within1 = sum(1 for a, b in zip(regex_scores, llm_scores) if abs(a-b)<=1) / len(all_results)
    pr, _ = pearsonr(regex_scores, llm_scores)
    
    output = {"n_judged": len(all_results), "within1": within1, "pearson_r": pr, "results": all_results}
    with open(expanded_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved: n={len(all_results)}, within1={within1:.1%}", flush=True)

if __name__ == "__main__":
    main()
