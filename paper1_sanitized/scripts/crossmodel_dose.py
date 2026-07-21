#!/usr/bin/env python3
"""
Cross-model dose-response: test dose gradient on multiple models.
Key question: does the LOGIC ceiling hold across models?
"""
import json, re, time, sys
from pathlib import Path
from collections import defaultdict
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent
client = OpenAI(api_key="<REDACTED_SECRET>", base_url="<REDACTED_URL>")
DELAY = 4.0

MODELS = ["deepseek-v4-0324", "claude-sonnet-4-20250514", "o4-mini"]

DOSES = {
    'EDIT': {
        'strong': "STOP. You must re-read the exact current contents of the target file using cat or a read command. Your previous edit failed because your text did not match the file exactly. Do not attempt any edit until you have refreshed your view of the file.",
    },
    'LOGIC': {
        'strong': "STOP. Your fix is logically incorrect. Analyze the failing test assertion carefully—what exact value does it expect? Your current approach produces the wrong result. Identify the specific computation error and fix only that.",
    },
}

def call_llm(prompt, model):
    time.sleep(DELAY)
    try:
        resp = client.chat.completions.create(
            model=model, messages=[{"role":"user","content":prompt}],
            max_tokens=1024, temperature=0)
        return resp.choices[0].message.content or ""
    except:
        time.sleep(8)
        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role":"user","content":prompt}],
                max_tokens=1024, temperature=0)
            return resp.choices[0].message.content or ""
        except:
            return ""

def evaluate(response, gold_files, ftype):
    if not response:
        return {"score": 0, "file_hit": False, "actionable": False, "relevant": False}
    file_hit = any(gf.split('/')[-1] in response for gf in gold_files)
    actionable = bool(re.search(r'(cat |grep |find |python |pytest |str_replace|def |import |sed )', response))
    relevant_patterns = {
        "EDIT": r'read|cat|view|exact|character|whitespace',
        "LOGIC": r'test|assert|expect|return|logic|fix|edge',
    }
    relevant = bool(re.search(relevant_patterns.get(ftype, ''), response, re.I))
    return {"score": sum([file_hit, actionable, relevant]), "file_hit": file_hit,
            "actionable": actionable, "relevant": relevant}

def main():
    annots = json.load(open(PROJECT_ROOT / "results/phase0_annotations/phase0_v2_annotations.json"))
    gold_map = {a['instance_id']: a['gold_files'] for a in annots['annotations']}
    
    out_path = PROJECT_ROOT / "results/crossmodel_dose.json"
    if out_path.exists():
        existing = json.load(open(out_path))
        done_keys = set(f"{r['instance_id']}_{r['model']}_{r['failure_type']}" for r in existing['results'])
        results = existing['results']
    else:
        done_keys = set()
        results = []
    
    type_instances = defaultdict(list)
    for a in annots['annotations']:
        if a['failure_type'] in ['EDIT', 'LOGIC']:
            type_instances[a['failure_type']].append(a)
    
    batch_done = 0
    MAX_BATCH = 8
    
    for model in MODELS:
        for ft in ['EDIT', 'LOGIC']:
            instances = type_instances[ft][:10]
            for inst in instances:
                iid = inst['instance_id']
                key = f"{iid}_{model}_{ft}"
                if key in done_keys:
                    continue
                if batch_done >= MAX_BATCH:
                    break
                
                gold_files = gold_map[iid]
                scaffold = DOSES[ft]['strong']
                
                prompt = f"""Expert engineer helping a failed code agent. Issue: {iid}, Failure: {ft}

Recovery guidance:
{scaffold}

Files to fix: {', '.join(gold_files)}

Provide the SINGLE BEST next action. Be specific."""
                
                response = call_llm(prompt, model)
                ev = evaluate(response, gold_files, ft)
                
                results.append({
                    "instance_id": iid,
                    "failure_type": ft,
                    "model": model,
                    "dose": "strong",
                    "eval": ev,
                    "response": response[:100]
                })
                done_keys.add(key)
                batch_done += 1
            if batch_done >= MAX_BATCH:
                break
        if batch_done >= MAX_BATCH:
            break
    
    with open(out_path, 'w') as f:
        json.dump({"n": len(results), "results": results}, f, indent=2)
    print(f"Saved: {len(results)} total ({batch_done} new)", flush=True)

if __name__ == "__main__":
    main()
