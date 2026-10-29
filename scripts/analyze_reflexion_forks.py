#!/usr/bin/env python3
"""Audit an absence-to-presence pilot without equating exposure with adoption."""
import argparse
import re
from pathlib import Path
from agent_reliability.io import digest,read_json,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    identity=read_json(a.run/'run.json')['identity'];target=identity['target'];rows=[]
    for path in sorted((a.run/'branches').glob('*/trial-*/result.json')):
        r=read_json(path);actions=read_json(path.parent/'checkpoint.json')['transitions']
        # A narrow, explicit action indicator. Visiting the fridge is also a
        # legitimate search action and is not itself evidence of false belief.
        direct=[i for i,x in enumerate(actions) if re.fullmatch(r'take tomato \d+ from fridge 1',x['action'])]
        rows.append({'branch':path.parent.parent.name,'trial':r['trial'],'success':r['success'],
            'decisions':r['decisions'],'initial_observation_sha256':r['initial_observation_sha256'],
            'trajectory_sha256':r['trajectory_sha256'],
            'world_commands_sha256':digest([x['submitted_action'] for x in actions if not x['action'].startswith('think:')]),
            'injected_clause_in_visible_memory':any(target['replacement_text'] in m for m in r['memory_before'][-3:]),
            'direct_target_take_actions':direct,
            'new_reflection_has_exact_injected_clause':len(r['memory_after'])>len(r['memory_before']) and target['replacement_text'] in r['memory_after'][-1],
            'fridge_visits':sum(x['action']=='go to fridge 1' for x in actions)})
    by={(r['branch'],r['trial']):r for r in rows};controls={}
    for branch in ['sham','correct_target','retain_error','discard_new']:
        common=[t for b,t in by if b=='original' and (branch,t) in by]
        controls[branch]={'matched_attempts':len(common),
            'equal_trajectory_hashes':sum(by[('original',t)]['trajectory_sha256']==by[(branch,t)]['trajectory_sha256'] for t in common),
            'equal_world_command_sequences':sum(by[('original',t)]['world_commands_sha256']==by[(branch,t)]['world_commands_sha256'] for t in common)}
    report={'scope':'one independent development task; action indicator is not a general semantic adoption evaluator',
            'task_id':target['task_id'],'target_sha256':target['sha256'],'attempts':rows,
            'trajectory_controls':controls,'completed_attempts':len(rows)}
    report['sha256']=digest(report);write_json(a.output,report)
    print({'completed_attempts':len(rows),'trajectory_controls':controls})

if __name__=='__main__': main()
