#!/usr/bin/env python3
"""
Multi-step scaffold simulation: 2-turn recovery.
For each instance, compare scaffold vs control over TWO consecutive actions:
  Turn 1: scaffold/control prompt → response_1
  Turn 2: given response_1 as context, ask for next action → response_2

This tests whether scaffold benefits persist or decay after the first step.
Measures: file_hit, actionable, direction (scaffold maintains direction better?).
"""
import json, os, re, time, random, sys
from pathlib import Path
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent
API_KEY = "<REDACTED_SECRET>"
BASE_URL = "<REDACTED_URL>"
MODEL = "gpt-4o-mini"
DELAY = 6.5

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def call_llm(prompt):
    time.sleep(DELAY)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": prompt}],
                max_tokens=512, temperature=0)
            return resp.choices[0].message.content or ""
        except Exception as e:
            if "429" in str(e):
                time.sleep(10)
            else:
                time.sleep(5)
    return ""

def evaluate(response, gold_files):
    if not response:
        return {"file_hit": False, "actionable": False, "has_edit": False}
    file_hit = any(gf.split('/')[-1] in response for gf in gold_files)
    actionable = bool(re.search(r'(cat |grep |find |python |pytest |str_replace|sed |cd )', response))
    has_edit = bool(re.search(r'str_replace|sed\s|patch\s|def |class |import ', response))
    return {"file_hit": file_hit, "actionable": actionable, "has_edit": has_edit}

def main():
    # Load data
    annots = json.load(open(PROJECT_ROOT / "results/phase0_annotations/phase0_v2_annotations.json"))
    from datasets import load_dataset
    ds = load_dataset('AlexCuadron/SWE-Bench-Verified-O1-reasoning-high-results', split='test')
    traj_map = {r['issue_name']: r['full_conversation_jsonl'] for r in ds}
    
    # Select instances: 10 EDIT + 10 LOC
    edit_instances = [a for a in annots['annotations'] if a['failure_type'] == 'EDIT' and a['instance_id'] in traj_map][:10]
    loc_instances = [a for a in annots['annotations'] if a['failure_type'] == 'LOC' and a['instance_id'] in traj_map][:10]
    instances = edit_instances + loc_instances
    
    # Load scaffolds
    scaffolds = {
        'EDIT': (PROJECT_ROOT / "prompts/scaffolding/EDIT_A_reread_file.txt").read_text().strip(),
        'LOC': (PROJECT_ROOT / "prompts/scaffolding/LOC_B_reread_issue.txt").read_text().strip(),
    }
    control_text = (PROJECT_ROOT / "prompts/scaffolding/CONTROL_no_scaffold.txt").read_text().strip()
    
    results = []
    print(f"Multi-step simulation: {len(instances)} instances × 2 conditions × 2 turns", flush=True)
    print(f"Expected API calls: ~{len(instances)*4}", flush=True)
    
    for idx, inst in enumerate(instances):
        iid = inst['instance_id']
        ft = inst['failure_type']
        gold_files = inst['gold_files']
        
        # Extract brief context from trajectory
        try:
            traj_data = json.loads(traj_map[iid])
            ctx_parts = []
            for turn in traj_data[:3]:
                resp = turn.get('response', {})
                choices = resp.get('choices', [])
                if choices and isinstance(choices[0], dict):
                    c = choices[0].get('message', {}).get('content', '')
                    if c: ctx_parts.append(f"[Agent]: {c[:200]}")
            context = '\n'.join(ctx_parts)[:1500]
        except:
            context = f"Agent working on {iid}"
        
        for condition in ['scaffold', 'control']:
            guidance = scaffolds[ft] if condition == 'scaffold' else control_text
            
            # Turn 1: same as phase3
            prompt1 = f"""Expert engineer helping a failed code agent. Issue: {iid}, Failure: {ft}
Agent trajectory:
{context}

Recovery guidance:
{guidance}

Files to fix: {', '.join(gold_files)}

Provide the SINGLE BEST next action (bash command or file edit). Be specific."""
            
            resp1 = call_llm(prompt1)
            eval1 = evaluate(resp1, gold_files)
            
            # Turn 2: given resp1, what's the next action?
            prompt2 = f"""You are continuing to fix issue {iid}. 
Your previous action was:
{resp1[:400]}

What is your NEXT action? Continue working toward the fix. 
Files to fix: {', '.join(gold_files)}

Provide the next specific action."""
            
            resp2 = call_llm(prompt2)
            eval2 = evaluate(resp2, gold_files)
            
            results.append({
                "instance_id": iid,
                "failure_type": ft,
                "condition": condition,
                "turn1": {"response": resp1[:200], "eval": eval1},
                "turn2": {"response": resp2[:200], "eval": eval2},
            })
        
        if (idx + 1) % 5 == 0:
            print(f"  {idx+1}/{len(instances)} done", flush=True)
            # Incremental save
            with open(PROJECT_ROOT / "results/multistep_simulation.json", 'w') as f:
                json.dump({"n": len(results)//2, "results": results}, f, indent=2)
    
    # Final analysis
    import numpy as np
    print("\n=== MULTI-STEP RESULTS ===", flush=True)
    
    for ft in ['EDIT', 'LOC']:
        for turn in ['turn1', 'turn2']:
            s_hits = [r[turn]['eval']['file_hit'] for r in results if r['condition']=='scaffold' and r['failure_type']==ft]
            c_hits = [r[turn]['eval']['file_hit'] for r in results if r['condition']=='control' and r['failure_type']==ft]
            print(f"  {ft} {turn}: scaffold file_hit={np.mean(s_hits):.0%} vs control={np.mean(c_hits):.0%}", flush=True)
    
    # Save final
    with open(PROJECT_ROOT / "results/multistep_simulation.json", 'w') as f:
        json.dump({"n": len(results)//2, "results": results}, f, indent=2)
    print(f"\nSaved {len(results)} results", flush=True)

if __name__ == "__main__":
    main()
