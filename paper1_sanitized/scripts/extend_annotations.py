#!/usr/bin/env python3
"""Auto-annotate more trajectories to find additional PLAN instances."""
import json, re, sys
from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).parent.parent

def classify_trajectory(issue_name, traj_data, gold_patch_files):
    """Apply hierarchical classification rules."""
    # Extract files edited by agent
    agent_files = set()
    edit_errors = 0
    
    try:
        turns = json.loads(traj_data) if isinstance(traj_data, str) else traj_data
    except:
        return None
    
    for turn in turns:
        resp = turn.get('response', {})
        choices = resp.get('choices', [])
        for choice in choices:
            msg = choice.get('message', {})
            content = msg.get('content', '') or ''
            
            # Find file references in str_replace_editor calls
            file_matches = re.findall(r'(?:path|file)["\s:=]+([/\w._-]+\.py)', content)
            agent_files.update(file_matches)
            
            # Count edit errors
            if 'No match found' in content or 'did not appear verbatim' in content:
                edit_errors += 1
    
    # Step 1: File overlap check
    agent_basenames = set(f.split('/')[-1] for f in agent_files)
    gold_basenames = set(f.split('/')[-1] for f in gold_patch_files)
    
    if not agent_basenames.intersection(gold_basenames):
        return 'LOC'
    
    # Step 2: Edit execution check
    if edit_errors >= 2:
        return 'EDIT'
    
    # Step 3: For PLAN vs LOGIC, need more context
    # PLAN: agent targets different module/approach than gold
    # LOGIC: agent targets same area but wrong fix
    # Heuristic: if agent edits many different files (>3) or the trajectory is very long,
    # more likely PLAN (broad misguided strategy)
    if len(agent_files) > 5:
        return 'PLAN'
    
    return 'LOGIC'

def main():
    annots = json.load(open(PROJECT_ROOT / "results/phase0_annotations/phase0_v2_annotations.json"))
    current_ids = set(a['instance_id'] for a in annots['annotations'])
    
    # Load dataset
    ds = load_dataset('AlexCuadron/SWE-Bench-Verified-O1-reasoning-high-results', split='test')
    
    # Load gold patches from SWE-bench
    swe = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
    gold_files_map = {}
    for item in swe:
        patch = item.get('patch', '')
        files = re.findall(r'diff --git a/(.+?) b/', patch)
        gold_files_map[item['instance_id']] = files
    
    new_plan = []
    new_all = []
    batch = 0
    
    for r in ds:
        iid = r['issue_name']
        if iid in current_ids:
            continue
        if iid not in gold_files_map:
            continue
        
        traj = r.get('full_conversation_jsonl', '')
        if not traj or len(traj) < 100:
            continue
        
        gold_files = gold_files_map[iid]
        ftype = classify_trajectory(iid, traj, gold_files)
        
        if ftype:
            entry = {
                "instance_id": iid,
                "failure_type": ftype,
                "gold_files": gold_files,
                "source": "auto_extended"
            }
            new_all.append(entry)
            if ftype == 'PLAN':
                new_plan.append(entry)
        
        batch += 1
        if batch >= 200:  # Process 200 at a time
            break
    
    print(f"Annotated {len(new_all)} new trajectories:", flush=True)
    from collections import Counter
    counts = Counter(e['failure_type'] for e in new_all)
    for ft, c in sorted(counts.items()):
        print(f"  {ft}: {c}", flush=True)
    
    print(f"\nNew PLAN instances found: {len(new_plan)}", flush=True)
    for p in new_plan[:10]:
        print(f"  {p['instance_id']}", flush=True)
    
    # Save extended annotations
    extended = annots.copy()
    extended['annotations'] = annots['annotations'] + new_all
    extended['n_total'] = len(extended['annotations'])
    extended['n_extended'] = len(new_all)
    
    with open(PROJECT_ROOT / "results/phase0_annotations/phase0_v3_extended.json", 'w') as f:
        json.dump(extended, f, indent=2)
    
    print(f"\nSaved extended annotations: {extended['n_total']} total", flush=True)

if __name__ == "__main__":
    main()
