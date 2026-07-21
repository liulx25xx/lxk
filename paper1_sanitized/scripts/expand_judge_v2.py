#!/usr/bin/env python3
"""Expanded LLM judge - simplified version with verbose output."""
import json, os, re, time, random, sys
from pathlib import Path
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent
API_KEY = "<REDACTED_SECRET>"
BASE_URL = "<REDACTED_URL>"
MODEL = "gpt-4o-mini"
DELAY = 6.5

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def judge_one(instance_id, gold_files, response):
    prompt = f"""Evaluate this code agent recovery response.

Issue: {instance_id}
Correct files: {', '.join(gold_files)}
Response: {response}

Rate (1=yes, 0=no):
1. FILE_RELEVANT: References correct files?
2. ACTIONABLE: Contains executable action?
3. PROGRESS: Moves toward fix?

Output ONLY: {{"file_relevant":X,"actionable":X,"progress":X,"total":X}}"""
    
    time.sleep(DELAY)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80, temperature=0,
        )
        text = resp.choices[0].message.content or ""
        match = re.search(r'\{[^}]+\}', text)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"  ERR: {e}", file=sys.stderr)
        time.sleep(10)
    return None

def main():
    phase3 = json.load(open(PROJECT_ROOT / "results/phase3_full_scaffold/full_results.json"))
    annots = json.load(open(PROJECT_ROOT / "results/phase0_annotations/phase0_v2_annotations.json"))
    gold_map = {a['instance_id']: a['gold_files'] for a in annots['annotations']}
    
    existing = json.load(open(PROJECT_ROOT / "results/llm_judge_validation.json"))
    judged_keys = set(f"{r['instance_id']}_{r['strategy']}" for r in existing['results'])
    
    remaining = [r for r in phase3['results'] 
                 if f"{r['instance_id']}_{r['strategy']}" not in judged_keys 
                 and r['instance_id'] in gold_map]
    
    random.seed(42)
    random.shuffle(remaining)
    to_judge = remaining[:140]
    
    print(f"Starting: {len(to_judge)} to judge", flush=True)
    
    new_results = []
    errors = 0
    for i, r in enumerate(to_judge):
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
        else:
            errors += 1
            if errors > 10:
                print(f"Too many errors ({errors}), stopping", flush=True)
                break
        
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(to_judge)} done, {len(new_results)} ok, {errors} err", flush=True)
    
    # Combine and save
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    from collections import defaultdict
    
    all_results = existing['results'] + new_results
    regex_scores = [r['regex_score'] for r in all_results]
    llm_scores = [r['llm_score'] for r in all_results]
    
    within1 = sum(1 for a, b in zip(regex_scores, llm_scores) if abs(a-b) <= 1) / len(all_results)
    pr, _ = pearsonr(regex_scores, llm_scores)
    
    # Condition ordering
    type_cond_llm = defaultdict(lambda: defaultdict(list))
    type_cond_regex = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        cond = 'control' if 'CONTROL' in r['strategy'] else 'scaffold'
        type_cond_llm[r['failure_type']][cond].append(r['llm_score'])
        type_cond_regex[r['failure_type']][cond].append(r['regex_score'])
    
    print(f"\n=== RESULTS (n={len(all_results)}, new={len(new_results)}) ===", flush=True)
    print(f"Within-1: {within1:.1%}, Pearson r: {pr:.3f}", flush=True)
    
    ordering_ok = 0
    for ft in ['EDIT', 'LOC', 'LOGIC', 'PLAN']:
        s_l = np.mean(type_cond_llm[ft]['scaffold']) if type_cond_llm[ft]['scaffold'] else 0
        c_l = np.mean(type_cond_llm[ft]['control']) if type_cond_llm[ft]['control'] else 0
        s_r = np.mean(type_cond_regex[ft]['scaffold']) if type_cond_regex[ft]['scaffold'] else 0
        c_r = np.mean(type_cond_regex[ft]['control']) if type_cond_regex[ft]['control'] else 0
        match = (s_r > c_r) == (s_l > c_l)
        if match: ordering_ok += 1
        print(f"  {ft:8} regex: s={s_r:.2f}>c={c_r:.2f}={s_r>c_r}  LLM: s={s_l:.2f}>c={c_l:.2f}={s_l>c_l} [{match}]", flush=True)
    
    print(f"Ordering: {ordering_ok}/4", flush=True)
    
    with open(PROJECT_ROOT / "results/llm_judge_expanded.json", 'w') as f:
        json.dump({"n_judged": len(all_results), "n_new": len(new_results),
                   "within1_agreement": within1, "pearson_r": pr,
                   "ordering_preserved": ordering_ok == 4, "results": all_results}, f, indent=2)
    print("Saved!", flush=True)

if __name__ == "__main__":
    main()
