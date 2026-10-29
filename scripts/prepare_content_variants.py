#!/usr/bin/env python3
"""Build reviewed local text edits and verify native first-request construction."""
from pathlib import Path
import argparse, re, time
from agent_reliability.io import digest, file_hash, read_json, write_json
from agent_reliability.reference import official_environment
from agent_reliability.reflexion import author_namespace
from prepare_four_attempt_protocol import checked
from prepare_shared_calibration import save
from prepare_content_intervention import PUBLIC, PRIVATE, MANIFEST, PARENT
from calibrate_successor_success import features

REGISTRY=PUBLIC/'content-intervention-registry-20261028.json'
SEEDS=[2180,2181]
CONDITIONS=['original','neutralized','paraphrase']


def paraphrase(text,index):
    if index==41:return 'and tried alternative methods such as the stoveburner or toaster if the microwave failed to work.'
    swaps=[('My mistake was not recognizing','I made the mistake of not recognizing'),
           ('I also failed to realize','I also did not recognize'),
           ('I should have','I ought to have'),('I should','I ought to'),
           ('I will','I intend to'),('I may need','I might need'),
           ('I was unsuccessful','My attempt was unsuccessful')]
    result=text
    for a,b in swaps:result=result.replace(a,b)
    assert result!=text and result.strip()
    return result


def build():
    started=time.process_time(); manifest=checked(MANIFEST)
    assert not (PRIVATE/'contract.json').exists(),'No changes after sampling begins'
    view=read_json(PRIVATE/'reading-view.json');assert file_hash(PRIVATE/'reading-view.json')==manifest['reading_file_sha256']
    manual=read_json(PRIVATE/'manual-edits.json');assert [x[0] for x in manual]==list(range(46))
    private=[];entries=[]
    for row,(i,sentence_index,neutral) in zip(manifest['selected'],manual):
        item=view[i];text=item['text'];spans=list(re.finditer(r'.+?(?:[.!?](?=\s+[A-Z])|$)',text,re.S))
        assert [m.group(0).strip() for m in spans]==re.split(r'(?<=[.!?])\s+(?=[A-Z])',text)
        span=spans[sentence_index];a,b=span.span()
        while a<b and text[a].isspace():a+=1
        if sentence_index==0:
            prefix=re.match(r'(?:New plan|Plan):\s*',text[a:b]);assert prefix is not None;a+=prefix.end()
        if i==41:
            a=text.index('and tried alternative methods',a,b)
            neutral='and noted that the earlier attempt did not establish a different heating procedure or its prerequisites.'
        focal=text[a:b];sham=paraphrase(focal,i)
        variants={'original':text,'neutralized':text[:a]+neutral+text[b:],'paraphrase':text[:a]+sham+text[b:]}
        assert len(set(variants.values()))==3
        assert all(v.startswith(text[:a]) and v.endswith(text[b:]) for v in variants.values())
        assert item['memory'][row['native_slot']]==text and digest(text)==row['native_text_sha256']
        memories={c:item['memory'][:row['native_slot']]+[v]+item['memory'][row['native_slot']+1:] for c,v in variants.items()}
        assert all(len(m)==len(row['state']) for m in memories.values())
        same_features=[features(row['state'],'seven') for c in CONDITIONS]
        assert all(x==row['seven_history'][-1] for x in same_features)
        # The only changed span contains no correct target-inventory-at-heating instruction.
        assert not re.search(r'\binventory\b.*\b(?:before|while)\b.*\bheat',focal,re.I)
        entry={**row,'edit':{'start':a,'end':b,'source_span_sha256':digest(focal),
               'neutral_replacement':neutral,'paraphrase_replacement':sham,'sentence_index':sentence_index},
               'variant_text_sha256':{c:digest(v) for c,v in variants.items()},
               'variant_memory_sha256':{c:digest(m) for c,m in memories.items()},
               'original_target_category_preserved':True,'target_category_review':'Single-coder semantic review; correct target-rule text outside focal span copied exactly',
               'additional_target_advice_remains':i not in [14,27,41]}
        entries.append(entry);private.append({'index':i,'focal_original':focal,'focal_neutralized':neutral,'focal_paraphrase':sham,
            'variants':variants,'memories':memories})
    write_json(PRIVATE/'variants.json',private)
    registry=save(REGISTRY,{'scope':'One separable local advice clause per reflection; other sentences remain intact',
        'manifest_sha256':manifest['sha256'],'manual_edits_sha256':file_hash(PRIVATE/'manual-edits.json'),
        'private_variants_sha256':file_hash(PRIVATE/'variants.json'),'entries':entries,'seeds':SEEDS,'conditions':CONDITIONS,
        'annotation':'Single Codex full-text and focal-pair semantic self-review; prior aggregate outcomes known; no independent human reviewer',
        'paraphrase_rule':'Equivalent first-person/modal wording at the same focal span, retaining its recommendation; not exact edit-distance matching',
        'invariance':'Same memory entry/slot/native flags, original target category and complete seven-feature history',
        'preparation_source_sha256':file_hash(Path(__file__)),'new_model_calls':0,'parameter_fits':0})
    print({'registry':registry['sha256'],'edits':46,'new_model_calls':0,'build_cpu_seconds':time.process_time()-started})


def interface():
    started=time.process_time();reg=checked(REGISTRY);items=read_json(PRIVATE/'variants.json')
    assert file_hash(PRIVATE/'variants.json')==reg['private_variants_sha256']
    resets=[];prompts=[];ns,_=author_namespace();demos=read_json(Path('third_party/reflexion/prompts/alfworld_3prompts.json'))
    wrapper=read_json(Path('configs/qwen3_8b_non_thinking_wrapper.json'))
    cached=None
    audit_path=PUBLIC/'content-intervention-interface-20261028.json'
    if audit_path.exists():
        old=checked(audit_path);old_prompts=read_json(PRIVATE/'interface-prompts.json')
        assert file_hash(PRIVATE/'interface-prompts.json')==old['private_prompt_file_sha256']
        if not (PRIVATE/'initial-reset-audit.json').exists():
            (PRIVATE/'initial-reset-audit.json').write_bytes(audit_path.read_bytes())
            write_json(PRIVATE/'initial-interface-prompts.json',old_prompts)
        cached={r['index']:r['raw_prompt'].rsplit('\nHere is the task:\n',1)[1][:-2] for r in old_prompts if r['condition']=='original'}
        assert len(cached)==46 and len(old['official_seeded_resets'])==92
        assert {(r['task_id'],r['seed']) for r in old['official_seeded_resets']}=={(r['task_id'],s) for r in reg['entries'] for s in SEEDS}
    class Captured(BaseException):pass
    captured=[]
    def intercept(prompt,**kwargs):captured.append(prompt);raise Captured()
    ns['llm']=intercept
    for r,item in zip(reg['entries'],items):
        observed=[]
        for seed in SEEDS:
            env=None if cached else official_environment(Path('data/alfworld'),r['task'],seed,Path('third_party/react'))
            try:
                if cached:ob=cached[r['index']]
                else:
                    observations,_=env.reset();ob='\n'.join(observations[0].split('\n\n')[1:])
                observed.append(digest(ob))
                assert digest(ob)==r['initial_observation_sha256']
                resets.append({'task_id':r['task_id'],'seed':seed,'observation_sha256':digest(ob)})
                if seed!=SEEDS[0]:continue
                key=next(v for k,v in ns['PREFIXES'].items() if r['task']['task_type'].startswith(k))
                base='Interact with a household to solve a task. Here are two examples.\n'+demos[f'react_{key}_1']+demos[f'react_{key}_0']
                for condition in CONDITIONS:
                    captured.clear();memory=item['memories'][condition]
                    try:ns['alfworld_run'](env,base,memory,to_print=False,ob=ob,model='text-davinci-003')
                    except Captured:pass
                    assert len(captured)==1
                    raw=str(ns['EnvironmentHistory'](base,ob,memory,[]))+'>'
                    assert captured[0]==raw and all(t.strip() in raw for t in memory)
                    for j,t in enumerate(memory):assert f'Trial {j}:\n{t.strip()}' in raw
                    prompts.append({'index':r['index'],'condition':condition,'raw_prompt':raw,
                        'wrapped_prompt':wrapper['prefix']+raw+wrapper['suffix']})
            finally:
                if env is not None:env.close()
        assert len(set(observed))==1
        if (r['index']+1)%10==0:print({'interface_tasks':r['index']+1,'new_model_calls':0},flush=True)
    assert len(resets)==92 and len(prompts)==138
    write_json(PRIVATE/'interface-prompts.json',prompts)
    audit=save(PUBLIC/'content-intervention-interface-20261028.json',{'registry_sha256':reg['sha256'],
        'official_seeded_resets':resets,'native_first_request_checks':len(prompts),
        'private_prompt_file_sha256':file_hash(PRIVATE/'interface-prompts.json'),
        'seeded_reset_cache_source_sha256':file_hash(PRIVATE/'initial-reset-audit.json') if cached else None,
        'source_sha256':file_hash(Path(__file__)),'checks_passed':True,'new_model_calls':0,
        'interface_cpu_seconds':time.process_time()-started})
    p=read_json(PRIVATE/'preparation.json');p.update(status='interface_checked_pending_token_review',registry_sha256=reg['sha256'],interface_sha256=audit['sha256'])
    write_json(PRIVATE/'preparation.json',p);print({'interface':audit['sha256'],'new_model_calls':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build','interface']);a=parser.parse_args()
    build() if a.action=='build' else interface()
