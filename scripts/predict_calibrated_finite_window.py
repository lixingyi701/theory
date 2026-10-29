#!/usr/bin/env python3
"""Exact candidate forecasts from a frozen one-attempt kernel; no model calls."""
import argparse
from functools import lru_cache
from pathlib import Path
from agent_reliability.io import digest,file_hash,read_json,write_json


def remaining(history):
    return 3-len(history)+max(i for i,v in enumerate(history) if v=='E')+1 if 'E' in history else 0


def restore(history,operation):
    value=list(history)
    if operation=='retain': return value
    if operation not in {'correct_latest','delete_latest'}: raise ValueError('Unknown operation')
    if 'E' not in value: return value
    index=max(i for i,v in enumerate(value) if v=='E')
    if operation=='correct_latest': value[index]='C'
    else: del value[index]
    return value


def bounds(kernels,budget,r):
    """A is semantically unresolved: choose error/absence at each future branch."""
    @lru_cache(None)
    def recurse(b,r):
        if b==0: return (0.,0.)
        k=kernels[bool(r)]; lower_error,upper_error=recurse(b-1,3)
        lower_absent,upper_absent=recurse(b-1,max(0,r-1))
        lower=k['S']+k['E']*lower_error+(k['C']+k['U'])*lower_absent+k['A']*min(lower_error,lower_absent)
        upper=k['S']+k['E']*upper_error+(k['C']+k['U'])*upper_absent+k['A']*max(upper_error,upper_absent)
        return lower,upper
    return recurse(budget,r)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve the candidate forecast version')
    source=read_json(Path('repro/reflexion-20261023/state-representation-presence-baseline.json'))
    kernels={bool(state):next(v['joint'] for v in source['one_attempt_forecasts'] if v['labels']==state) for state in [[],['E']]}
    starts=[['E'],['N','E','N'],['E','N','E'],['E','E','N']]; rows=[]
    for history in starts:
        for budget in range(1,9):
            operations={}
            for operation in ['retain','correct_latest','delete_latest']:
                after=restore(history,operation); interval=bounds(kernels,budget,remaining(after))
                operations[operation]={'memory_after':after,'remaining_error_updates':remaining(after),
                    'success_probability_bounds':list(interval)}
            correct=operations['correct_latest']['success_probability_bounds']; delete=operations['delete_latest']['success_probability_bounds']
            rows.append({'initial_memory':history,'attempt_budget':budget,'operations':operations,
                'correct_minus_delete_bounds':[correct[0]-delete[1],correct[1]-delete[0]]})
    parameters={}
    for present,k in kernels.items():
        failure=1-k['S']
        parameters['error_present' if present else 'error_absent']={'joint':k,
            'failure_conditional_explicit_E':k['E']/failure,
            'failure_conditional_error_bounds_including_A':[k['E']/failure,(k['E']+k['A'])/failure]}
    value={'scope':'Conditional exact finite-window forecasts from a real-Agent-calibrated candidate kernel; not new Agent measurements',
        'source_presence_freeze_sha256':source['sha256'],'visible_window':3,'parameters':parameters,
        'initial_history':'Exactly the listed 1-3 entries; no older entries outside the visible window',
        'recovery':'One intervention before the first attempt; later attempts use native append-and-read-last-three updates; no repeated deletion exposing hidden old history',
        'assumptions':['Joint single-attempt probabilities depend only on target-error presence, not age, other text, or multiple-error/correction conflicts',
            'The same kernel remains valid after native reflection updates',
            'E denotes an error-containing entry; C and U do not contain the target error; A may or may not contain it'],
        'bounds_meaning':'Only semantic ambiguity of A is bounded; parameter-estimation error and state-compression error are not covered',
        'rows':rows,'model_calls':0,'driver_sha256':file_hash(Path(__file__))}
    value['sha256']=digest(value);write_json(args.output,value)
    print({'sha256':value['sha256'],'conditions':len(rows),'model_calls':0,
           'representative':[v for v in rows if v['initial_memory'] in [['E'],['E','N','E']] and v['attempt_budget'] in [1,2,3]]})


if __name__=='__main__':main()
