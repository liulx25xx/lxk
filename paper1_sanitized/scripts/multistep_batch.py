#!/usr/bin/env python3
"""Multistep simulation - small batch (3 instances) to fit in timeout."""
import json, re, time, sys
from pathlib import Path
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent
client = OpenAI(api_key="<REDACTED_SECRET>", base_url="<REDACTED_URL>")

def call_llm(prompt):
    time.sleep(6.5)
    try:
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=512, temperature=0)
        return resp.choices[0].message.content or ""
    except:
        time.sleep(8)
        try:
            resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=512, temperature=0)
            return resp.choices[0].message.content or ""
        except:
            return ""

def evaluate(response, gold_files):
    if not response: return {"file_hit":False,"actionable":False,"has_edit":False}
    file_hit = any(gf.split("/")[-1] in response for gf in gold_files)
    actionable = bool(re.search(r"(cat |grep |find |python |pytest |str_replace|sed )", response))
    has_edit = bool(re.search(r"str_replace|sed |def |class |import ", response))
    return {"file_hit":file_hit,"actionable":actionable,"has_edit":has_edit}

def main():
    annots = json.load(open(PROJECT_ROOT / "results/phase0_annotations/phase0_v2_annotations.json"))
    phase3 = json.load(open(PROJECT_ROOT / "results/phase3_full_scaffold/full_results.json"))
    phase3_instances = set(r["instance_id"] for r in phase3["results"])
    
    # Load existing results
    out_path = PROJECT_ROOT / "results/multistep_simulation.json"
    if out_path.exists():
        existing = json.load(open(out_path))
        done_ids = set(r["instance_id"] for r in existing["results"])
        results = existing["results"]
    else:
        done_ids = set()
        results = []
    
    # Get next batch of 3 instances not yet done
    all_candidates = [a for a in annots["annotations"] 
                      if a["failure_type"] in ("EDIT","LOC") 
                      and a["instance_id"] in phase3_instances
                      and a["instance_id"] not in done_ids]
    batch = all_candidates[:2]
    
    if not batch:
        print("All done!", flush=True)
        return
    
    scaffolds = {
        "EDIT": (PROJECT_ROOT / "prompts/scaffolding/EDIT_A_reread_file.txt").read_text().strip(),
        "LOC": (PROJECT_ROOT / "prompts/scaffolding/LOC_B_reread_issue.txt").read_text().strip(),
    }
    control_text = (PROJECT_ROOT / "prompts/scaffolding/CONTROL_no_scaffold.txt").read_text().strip()
    
    print(f"Multistep batch: {len(batch)} instances (have {len(done_ids)} done)", flush=True)
    
    for idx, inst in enumerate(batch):
        iid = inst["instance_id"]
        ft = inst["failure_type"]
        gold_files = inst["gold_files"]
        
        for condition in ["scaffold", "control"]:
            guidance = scaffolds[ft] if condition == "scaffold" else control_text
            
            prompt1 = f"Expert engineer helping failed code agent. Issue: {iid}, Type: {ft}\nGuidance: {guidance}\nFiles: {', '.join(gold_files)}\nProvide the SINGLE BEST next action. Be specific."
            resp1 = call_llm(prompt1)
            eval1 = evaluate(resp1, gold_files)
            
            prompt2 = f"Continue fixing {iid}. Previous action: {resp1[:300]}\nNext action? Files: {', '.join(gold_files)}"
            resp2 = call_llm(prompt2)
            eval2 = evaluate(resp2, gold_files)
            
            results.append({"instance_id":iid,"failure_type":ft,"condition":condition,
                           "turn1":{"response":resp1[:150],"eval":eval1},
                           "turn2":{"response":resp2[:150],"eval":eval2}})
        
        print(f"  {idx+1}/{len(batch)} done ({iid})", flush=True)
        
        # Save immediately after each instance
        n_instances = len(set(r["instance_id"] for r in results))
        with open(out_path, "w") as f:
            json.dump({"n": n_instances, "results": results}, f, indent=2)
    
    # Save
    n_instances = len(set(r["instance_id"] for r in results))
    with open(out_path, "w") as f:
        json.dump({"n": n_instances, "results": results}, f, indent=2)
    print(f"Saved: {n_instances} instances total", flush=True)

if __name__ == "__main__":
    main()
