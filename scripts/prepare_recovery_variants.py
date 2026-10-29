#!/usr/bin/env python3
"""Prepare reviewed native-error interventions, interface and precision; zero inference."""
from pathlib import Path
import argparse,time,statistics,math
from collections import defaultdict
from agent_reliability.io import digest,file_hash,read_json,write_json
from agent_reliability.reference import official_environment
from agent_reliability.reflexion import author_namespace
from prepare_four_attempt_protocol import checked
from prepare_shared_calibration import save
from diagnose_second_attempt import raw_index,read_attempt

PUBLIC=Path('repro/reflexion-20261025'); ROOT=Path('runs/reflexion-recovery-mechanism-20261029')
REGISTRY=PUBLIC/'recovery-mechanism-registry-20261029.json'
CONDITIONS=['original','paraphrase','correct','remove_target','delete_entry']
SEEDS=[2190,2191]


def build():
    assert not (ROOT/'contract.json').exists() and not REGISTRY.exists(),'Preserve intervention preparation'
    manifest=checked(PUBLIC/'recovery-mechanism-parents-20261029.json')
    view=read_json(ROOT/'parent-reading.json');assert file_hash(ROOT/'parent-reading.json')==manifest['private_reading_sha256']
    manual=read_json(ROOT/'manual-edits.json'); assert [r['index'] for r in manual]==list(range(51))
    raw,_=raw_index(); inputs=read_json(Path('runs/reflexion-seven-parameter-development-20261027/input-events.json'))
    by_id={(r['trajectory_id'],r['trial']):r for r in inputs}; public=[]; private=[]
    for case,row,edits in zip(manifest['cases'],view,manual):
        source=row['native_text'];memory=row['memory'];assert digest(source)==case['native_text_sha256']
        assert digest(memory)==case['memory_sha256'] and len(memory)==3 and memory[-1]==source
        old=by_id[tuple(case['parent_failure_id'])];_,_,result,cp,identity=read_attempt(old,raw)
        assert not result['success'] and result['memory_after'][-3:]==memory
        variants={'original':source}
        for c,key in [('paraphrase','paraphrase'),('correct','correct'),('remove_target','remove')]:
            t=source; end=len(source)
            for e in reversed(edits['edits']):
                a,b=e['start'],e['end'];assert 0<=a<b<=end and source[a:b]==e['original'];end=a
                t=t[:a]+e[key]+t[b:]
            assert t.startswith('Plan:') and t.strip();variants[c]=t
        assert len(set(variants.values()))==4
        memories={c:memory[:-1]+[t] for c,t in variants.items()};memories['delete_entry']=memory[:-1]
        assert len(memories['delete_entry'])==2
        entry=dict(case,task=identity['task'],edits=edits['edits'],
            memory_hashes={c:digest(m) for c,m in memories.items()},
            text_hashes={c:digest(t) for c,t in variants.items()},
            scope='All reviewed target-microwave-placement spans; other appliance, occupancy and failure hypotheses remain. Sentence spans can include collateral commands or context.',
            initial_native_count={c:int(c!='delete_entry') for c in CONDITIONS})
        public.append(entry);private.append(dict(index=case['index'],variants=variants,memories=memories))
    write_json(ROOT/'variants.json',private)
    r=save(REGISTRY,dict(parents_sha256=manifest['sha256'],entries=public,conditions=CONDITIONS,seeds=SEEDS,
        private_variants_sha256=file_hash(ROOT/'variants.json'),manual_edits_sha256=file_hash(ROOT/'manual-edits.json'),
        source_sha256=file_hash(Path(__file__)),edited_spans=sum(len(r['edits']) for r in manual),
        annotation='Single Codex full-text reading and paired self-review; historical aggregates known; no independent human/blind review.',
        new_model_calls=0,parameter_fits=0))
    print(dict(registry=r['sha256'],parents=51,spans=r['edited_spans']),flush=True)


def interface(model_dir):
    import subprocess,json,os
    from scipy.stats import t
    assert not (ROOT/'contract.json').exists()
    reg=checked(REGISTRY);vv=read_json(ROOT/'variants.json');assert file_hash(ROOT/'variants.json')==reg['private_variants_sha256']
    assert not (PUBLIC/'recovery-mechanism-interface-20261029.json').exists(),'Preserve prepared interface'
    ns,_=author_namespace(); demos=read_json(Path('third_party/reflexion/prompts/alfworld_3prompts.json'))
    wrapper=read_json(Path('configs/qwen3_8b_non_thinking_wrapper.json'))
    class Captured(BaseException):pass
    captured=[]
    def capture(prompt,**kwargs):captured.append(prompt);raise Captured()
    ns['llm']=capture;prompts=[];resets=[];tokens=[]
    for c,v in zip(reg['entries'],vv):
        for seed in SEEDS:
            env=official_environment(Path('data/alfworld'),c['task'],seed,Path('third_party/react'))
            try:
                ob='\n'.join(env.reset()[0][0].split('\n\n')[1:]);assert digest(ob)==c['initial_observation_sha256']
                resets.append(dict(task_id=c['task_id'],seed=seed,observation_sha256=digest(ob)))
                if seed!=SEEDS[0]:continue
                k=next(v for k,v in ns['PREFIXES'].items() if c['task']['task_type'].startswith(k))
                base='Interact with a household to solve a task. Here are two examples.\n'+demos[f'react_{k}_1']+demos[f'react_{k}_0']
                counts={}; hashes={}
                for cond in CONDITIONS:
                    memory=v['memories'][cond];captured.clear()
                    try:ns['alfworld_run'](env,base,memory,to_print=False,ob=ob,model='text-davinci-003')
                    except Captured:pass
                    raw=str(ns['EnvironmentHistory'](base,ob,memory,[]))+'>'
                    assert captured==[raw]
                    assert all(f'Trial {j}:\n{m.strip()}' in raw for j,m in enumerate(memory))
                    wrapped=wrapper['prefix']+raw+wrapper['suffix'];n=0
                    counts[cond]=n;hashes[cond]=digest(wrapped)
                    prompts.append(dict(index=c['index'],condition=cond,raw_prompt=raw,wrapped_prompt=wrapped))
                tokens.append(dict(index=c['index'],first_prompt_tokens=counts,first_prompt_sha256=hashes,
                    correct_minus_paraphrase=counts['correct']-counts['paraphrase']))
            finally:env.close()
        if (c['index']+1)%10==0:print(dict(interface_tasks=c['index']+1,new_model_calls=0),flush=True)
    assert len(resets)==102 and len(prompts)==255
    write_json(ROOT/'interface-prompts.json',prompts)
    # Use the existing inference environment's CPU tokenizer without installing it in runner.
    code='from tokenizers import Tokenizer;import json,sys;from pathlib import Path;t=Tokenizer.from_file(sys.argv[1]);p=json.loads(Path(sys.argv[2]).read_text());print(json.dumps([len(t.encode(r["wrapped_prompt"],add_special_tokens=False).ids) for r in p]))'
    completed=subprocess.run([os.environ['THEORY_TOKENIZER_PYTHON'],'-c',code,str(model_dir/'tokenizer.json'),str(ROOT/'interface-prompts.json')],capture_output=True,text=True,check=True,timeout=120)
    sizes=json.loads(completed.stdout);assert len(sizes)==255 and max(sizes)+256<=32768
    lookup={(r['index'],r['condition']):n for r,n in zip(prompts,sizes)}
    for r in tokens:
        r['first_prompt_tokens']={c:lookup[r['index'],c] for c in CONDITIONS}
        r['correct_minus_paraphrase']=lookup[r['index'],'correct']-lookup[r['index'],'paraphrase']
    native=checked(PUBLIC/'native-clause-results.json'); a=[r['mean'] for r in native['actor_success_contrasts']['C_minus_E']['task_effects']]
    old=checked(PUBLIC/'stage-calibration-results-20261029.json')['recovery_curves']['rates']
    d=[old['correct_latest']['scene_curves'][s][3]-old['delete_latest']['scene_curves'][s][3] for s in old['correct_latest']['scene_curves']]
    critical=float(t.ppf(.9875,22));planning={}
    for name,values in [('prior_native_immediate_C_E',a),('prior_controlled_budget4_C_delete',d)]:
        sd=statistics.stdev(values);planning[name]=dict(source_units=len(values),source_sd=sd,projected_97_5_halfwidth_23_scenes=critical*sd/math.sqrt(23),
            interpretation='Scenario using older, different panels; not guaranteed precision or claimed power. Two-point policy differences are not a design target.')
    save(PUBLIC/'recovery-mechanism-interface-20261029.json',dict(registry_sha256=reg['sha256'],official_seeded_resets=resets,
        native_first_request_checks=255,private_prompt_file_sha256=file_hash(ROOT/'interface-prompts.json'),tokenizer_sha256=file_hash(model_dir/'tokenizer.json'),
        tokens=tokens,correct_paraphrase_token_difference_range=[min(r['correct_minus_paraphrase'] for r in tokens),max(r['correct_minus_paraphrase'] for r in tokens)],
        token_scope='Lengths recorded, no artificial neutral padding or exact token matching; text editing effects include these differences.',
        primary_t97_5_critical_df22=critical,secondary_t95_critical_df22=float(t.ppf(.975,22)),precision_scenarios=planning,
        source_sha256=file_hash(Path(__file__)),checks_passed=True,new_model_calls=0))
    prep=read_json(ROOT/'preparation.json');prep.update(status='variants_and_interface_complete_pending_protocol_freeze',registry_sha256=reg['sha256']);write_json(ROOT/'preparation.json',prep)
    print(dict(official_resets=102,prompts=255,precision=planning,new_model_calls=0),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','interface']);p.add_argument('--model-dir',type=Path);a=p.parse_args()
    build() if a.action=='build' else interface(a.model_dir)
