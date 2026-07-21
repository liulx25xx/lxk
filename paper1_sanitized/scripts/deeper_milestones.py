#!/usr/bin/env python3
"""Analyze deeper milestones from existing response data (no API calls needed)."""
import json, re, numpy as np
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
phase3 = json.load(open(PROJECT_ROOT / "results/phase3_full_scaffold/full_results.json"))

def has_edit_attempt(response):
    return bool(re.search(r'str_replace|sed\s|patch\s|>>>.*<<<|diff|cat.*>>', response))

def has_valid_command(response):
    if not response or len(response) < 5:
        return False
    if response.rstrip().endswith('...') or len(response) >= 148:
        return False
    return bool(re.search(r'```|^\$\s|^(cat|grep|find|python|pytest|cd|sed)', response, re.M))

# Compute milestones per type x condition
type_cond = defaultdict(lambda: defaultdict(lambda: {'file_hit': [], 'edit_attempt': [], 'valid_cmd': []}))

for r in phase3['results']:
    ft = r['failure_type']
    is_ctrl = 'CONTROL' in r['strategy']
    cond = 'control' if is_ctrl else 'scaffold'
    resp = r.get('response', '')
    
    type_cond[ft][cond]['file_hit'].append(int(r['eval']['file_hit']))
    type_cond[ft][cond]['edit_attempt'].append(int(has_edit_attempt(resp)))
    type_cond[ft][cond]['valid_cmd'].append(int(has_valid_command(resp)))

print('DEEPER MILESTONES BY TYPE (scaffold vs control)')
print('='*70)
print(f'{"Type":<8} {"Metric":<15} {"Scaffold":>10} {"Control":>10} {"Delta":>8}')
print('-'*55)

for ft in ['EDIT', 'LOC', 'LOGIC', 'PLAN']:
    for metric in ['file_hit', 'edit_attempt', 'valid_cmd']:
        s = np.mean(type_cond[ft]['scaffold'][metric])
        c = np.mean(type_cond[ft]['control'][metric])
        print(f'{ft:<8} {metric:<15} {s:>10.1%} {c:>10.1%} {s-c:>+8.1%}')
    print()

# LOC mismatch on deeper metrics
print('\nLOC MISMATCH (test_guided) deeper milestones:')
loc_tg = {'file_hit': [], 'edit_attempt': [], 'valid_cmd': []}
loc_ctrl = {'file_hit': [], 'edit_attempt': [], 'valid_cmd': []}
for r in phase3['results']:
    if r['failure_type'] != 'LOC':
        continue
    resp = r.get('response', '')
    d = {'file_hit': int(r['eval']['file_hit']),
         'edit_attempt': int(has_edit_attempt(resp)),
         'valid_cmd': int(has_valid_command(resp))}
    if r['strategy'] == 'LOC_C_test_guided':
        for k, v in d.items():
            loc_tg[k].append(v)
    elif r['strategy'] == 'CONTROL_no_scaffold':
        for k, v in d.items():
            loc_ctrl[k].append(v)

for metric in ['file_hit', 'edit_attempt', 'valid_cmd']:
    tg = np.mean(loc_tg[metric])
    ct = np.mean(loc_ctrl[metric])
    print(f'  {metric:<15} test_guided={tg:.1%}  control={ct:.1%}  delta={tg-ct:+.1%}')
