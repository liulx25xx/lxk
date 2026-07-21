#!/usr/bin/env python3
"""
Expanded LLM-as-Judge validation (blinded).
Judge does NOT know the strategy type — only sees response + gold files + failure context.
Uses a simplified 0-3 rubric without type-specific keywords.
"""
import json, os, re, time, random
from pathlib import Path
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent
API_KEY = os.environ.get("OPENAI_API_KEY", "<REDACTED_SECRET>")
BASE_URL = "<REDACTED_URL>"
MODEL = "gpt-4o-mini"
DELAY = 6.5
TARGET_NEW = 140

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

JUDGE_PROMPT = """You are evaluating a code agent's recovery response after a failed attempt.

The agent was working on issue: {instance_id}
The correct files to modify are: {gold_files}
The agent's previous attempt failed. It was given guidance and produced this response:

---RESPONSE---
{response}
---END RESPONSE---

Rate the response on THREE dimensions (1 = yes, 0 = no):

1. FILE_RELEVANT: Does the response reference or operate on any of the correct files listed above?
2. ACTIONABLE: Does the response contain a concrete, executable next action (a command, code edit, or specific file operation)?
3. PROGRESS: Based on the response alone, would executing it likely bring the agent closer to fixing the issue (correct direction, reasonable approach)?

Output ONLY a JSON object: {{"file_relevant": 0 or 1, "actionable": 0 or 1, "progress": 0 or 1, "total": sum}}
"""


def judge_response(instance_id, gold_files, response):
    prompt = JUDGE_PROMPT.format(
        instance_id=instance_id,
        gold_files=", ".join(gold_files),
        response=response[:500]
    )
    for attempt in range(3):
        time.sleep(DELAY)
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100, temperature=0,
            )
            text = resp.choices[0].message.content or ""
            # Parse JSON from response
            match = re.search(r'\{[^}]+\}', text)
            if match:
                data = json.loads(match.group())
                return data
        except Exception as e:
            if "429" in str(e):
                time.sleep(10 * (attempt + 1))
            else:
                print(f"  Error (attempt {attempt+1}): {e}")
                time.sleep(5)
    return None


def main():
    # Load existing judge results
    judge_path = PROJECT_ROOT / "results" / "llm_judge_validation.json"
    existing = json.load(open(judge_path))
    judged_keys = set()
    for r in existing['results']:
        key = f"{r['instance_id']}_{r['strategy']}"
        judged_keys.add(key)
    
    # Load phase3 results
    phase3 = json.load(open(PROJECT_ROOT / "results" / "phase3_full_scaffold" / "full_results.json"))
    
    # Load gold files
    annots = json.load(open(PROJECT_ROOT / "results" / "phase0_annotations" / "phase0_v2_annotations.json"))
    gold_map = {a['instance_id']: a['gold_files'] for a in annots['annotations']}
    
    # Select remaining samples (stratified by type and strategy)
    remaining = []
    for r in phase3['results']:
        key = f"{r['instance_id']}_{r['strategy']}"
        if key not in judged_keys and r['instance_id'] in gold_map:
            remaining.append(r)
    
    random.seed(42)
    random.shuffle(remaining)
    to_judge = remaining[:TARGET_NEW]
    
    print(f"Judging {len(to_judge)} new responses (blinded, no type info to judge)")
    print(f"Already have {len(existing['results'])} judged")
    
    new_results = []
    for i, r in enumerate(to_judge):
        iid = r['instance_id']
        gold_files = gold_map.get(iid, [])
        response = r.get('response', '')
        
        # Get full response if truncated
        if len(response) < 20:
            continue
            
        result = judge_response(iid, gold_files, response)
        if result is None:
            continue
        
        entry = {
            "instance_id": iid,
            "failure_type": r['failure_type'],
            "strategy": r['strategy'],
            "regex_eval": r['eval'],
            "regex_score": r['eval']['score'],
            "llm_eval": result,
            "llm_score": result.get('total', 0)
        }
        new_results.append(entry)
        
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(to_judge)} done")
    
    # Combine with existing
    all_results = existing['results'] + new_results
    
    # Compute stats
    import numpy as np
    regex_scores = [r['regex_score'] for r in all_results]
    llm_scores = [r['llm_score'] for r in all_results]
    
    exact = sum(1 for a, b in zip(regex_scores, llm_scores) if a == b) / len(all_results)
    within1 = sum(1 for a, b in zip(regex_scores, llm_scores) if abs(a - b) <= 1) / len(all_results)
    
    from scipy.stats import pearsonr, spearmanr
    pr, _ = pearsonr(regex_scores, llm_scores)
    sr, _ = spearmanr(regex_scores, llm_scores)
    
    # Condition-level ordering check
    from collections import defaultdict
    type_cond_regex = defaultdict(lambda: defaultdict(list))
    type_cond_llm = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        ft = r['failure_type']
        is_ctrl = 'CONTROL' in r['strategy']
        cond = 'control' if is_ctrl else 'scaffold'
        type_cond_regex[ft][cond].append(r['regex_score'])
        type_cond_llm[ft][cond].append(r['llm_score'])
    
    print(f"\n=== EXPANDED JUDGE RESULTS (n={len(all_results)}) ===")
    print(f"Exact agreement: {exact:.1%}")
    print(f"Within-1 agreement: {within1:.1%}")
    print(f"Pearson r: {pr:.3f}")
    print(f"Spearman rho: {sr:.3f}")
    
    print(f"\nCondition ordering preserved?")
    ordering_ok = 0
    ordering_total = 0
    for ft in ['EDIT', 'LOC', 'LOGIC', 'PLAN']:
        if type_cond_llm[ft]['scaffold'] and type_cond_llm[ft]['control']:
            s_llm = np.mean(type_cond_llm[ft]['scaffold'])
            c_llm = np.mean(type_cond_llm[ft]['control'])
            s_regex = np.mean(type_cond_regex[ft]['scaffold'])
            c_regex = np.mean(type_cond_regex[ft]['control'])
            
            regex_order = s_regex > c_regex
            llm_order = s_llm > c_llm
            match = "YES" if regex_order == llm_order else "NO"
            if regex_order == llm_order:
                ordering_ok += 1
            ordering_total += 1
            
            print(f"  {ft:8} regex: scaffold={s_regex:.2f} > ctrl={c_regex:.2f}={regex_order}, "
                  f"LLM: scaffold={s_llm:.2f} > ctrl={c_llm:.2f}={llm_order} [{match}]")
    
    print(f"\n  Ordering agreement: {ordering_ok}/{ordering_total}")
    
    # Save expanded results
    output = {
        "n_judged": len(all_results),
        "n_original": len(existing['results']),
        "n_new": len(new_results),
        "pearson_r": pr,
        "spearman_rho": sr,
        "exact_agreement": exact,
        "within1_agreement": within1,
        "ordering_preserved": ordering_ok == ordering_total,
        "results": all_results
    }
    
    out_path = PROJECT_ROOT / "results" / "llm_judge_expanded.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
