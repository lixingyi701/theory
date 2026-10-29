#!/usr/bin/env python3
"""Bounded same-parent local advice intervention on one idle GPU replica."""
from pathlib import Path
from collections import Counter
import argparse, time
from agent_reliability.cli import provenance
from agent_reliability.io import digest, file_hash, read_json, write_json, run_lock
from agent_reliability.reference import CompletionTransport, ACTION_ALIAS_PROFILE
from agent_reliability.reflexion import run_attempt, author_namespace
from prepare_shared_calibration import save
from prepare_four_attempt_protocol import checked
from prepare_content_intervention import PUBLIC, PRIVATE, MANIFEST
from prepare_content_variants import REGISTRY, SEEDS, CONDITIONS
from run_middle_context_transport import reserve_call, replica_audit, CalibrationStop
from continue_reflexion_pilot import ledger
from summarize_reflexion import costs

PROTOCOL=PUBLIC/'content-intervention-protocol-20261028.json'
LIMITS={'additional_calls_including_replica_checks':25000,'seconds':43200}
T_CRITICAL=2.06865761041904  # df23, two-sided95%;24 retained scenes.


def matrix(registry):
    rows=[]
    for r in registry['entries']:
        for j,seed in enumerate(SEEDS):
            order=CONDITIONS[(r['index']+j)%3:]+CONDITIONS[:(r['index']+j)%3]
            if (r['index']+j)%2:order=list(reversed(order))
            for condition in order:
                rows.append({'id':f'content-{r["task_id"]}-{seed}-{condition}','task_id':r['task_id'],
                    'scene_id':r['scene_id'],'index':r['index'],'condition':condition,'seed':seed,'trial':2,
                    'memory_sha256':r['variant_memory_sha256'][condition]})
    return rows


def inputs():
    registry=checked(REGISTRY);variants=read_json(PRIVATE/'variants.json')
    assert file_hash(PRIVATE/'variants.json')==registry['private_variants_sha256']
    for r,v in zip(registry['entries'],variants):
        for c in CONDITIONS:
            assert digest(v['memories'][c])==r['variant_memory_sha256'][c]
            assert digest(v['variants'][c])==r['variant_text_sha256'][c]
            original=v['variants']['original'];e=r['edit'];a,b=e['start'],e['end']
            replacement=original[a:b] if c=='original' else e['neutral_replacement' if c=='neutralized' else 'paraphrase_replacement']
            assert v['variants'][c]==original[:a]+replacement+original[b:]
            assert digest(original[a:b])==e['source_span_sha256']
    return registry,variants


def prepare():
    assert not PROTOCOL.exists() and not (PRIVATE/'contract.json').exists(),'Do not reset collection'
    reg,_=inputs();interface=checked(PUBLIC/'content-intervention-interface-20261028.json')
    tokens=checked(PUBLIC/'content-intervention-tokens-20261028.json')
    assert interface['registry_sha256']==tokens['registry_sha256']==reg['sha256']
    assert interface['checks_passed'] and tokens['checks_passed']
    assert len(reg['entries'])==46 and len({r['scene_id'] for r in reg['entries']})==24
    calls=ledger();assert sum(v['dispatched'] for v in calls.values())==sum(v['responses'] for v in calls.values())==200396
    protected=checked(PUBLIC/'second-attempt-protocol-20261028.json')['protected_files'].copy()
    names=['scripts/prepare_content_intervention.py','scripts/prepare_content_variants.py','scripts/audit_content_tokens.py',
        'scripts/run_content_intervention.py','scripts/evaluate_content_intervention.py','scripts/run_content_campaign.sh',
        'docs/experiments/content-intervention-design-20261028.md',str(MANIFEST),str(REGISTRY),
        str(PUBLIC/'content-intervention-interface-20261028.json'),str(PUBLIC/'content-intervention-tokens-20261028.json'),
        'src/agent_reliability/reflexion.py','third_party/reflexion/env_history.py','third_party/reflexion/alfworld_trial.py']
    for name in names:protected[name]=file_hash(Path(name))
    p=save(PROTOCOL,{'scope':'Prospective paired local-clause intervention on reused tasks, testing seven-state invariance',
        'registry_sha256':reg['sha256'],'manifest_sha256':checked(MANIFEST)['sha256'],
        'interface_sha256':interface['sha256'],'tokens_sha256':tokens['sha256'],
        'cases':reg['entries'],'cells':matrix(reg),'seeds':SEEDS,'conditions':CONDITIONS,
        'tasks':46,'scenes':24,'expected_attempts':276,'max_actor_decisions':49,'trial_number':2,
        'ports':[8004],'gpu_indices':[3],'gpu_memory_utilization':.35,'limits':LIMITS,'prior_global_calls':200396,
        'source_hashes':provenance()['source_hashes'],'protected_files':protected,
        'input_format':'qwen3_non_thinking_actionline','decoding_profile':'qwen3_non_thinking_recommended_v1',
        'reflector':'Unchanged native update after failure; new reflection not used for a further actor attempt',
        'analysis':{'primary':'neutralized-minus-paraphrase native success fraction',
            'aggregation':'Average paired seeds within task, tasks within scene, then24 scenes equally',
            'interval':'Paired scene t95, df23','t_critical':T_CRITICAL,
            'secondary':['neutralized-minus-original','paraphrase-minus-original','confirmed target heating','put-then-heat order','physical no-effect actions','decisions and calls'],
            'bootstrap_replicates':20000,'bootstrap_seed':2026102802,
            'decision':'Report difference/interval; positive lower primary bound supports benefit of local neutralization. No beta promotion, automatic eighth input or retroactive budget-validation claim.'},
        'interpretation':'Only one focal separable advice clause is changed; other advice may remain. A null result does not establish full-memory sufficiency.',
        'holdout134_calls':0,'parameter_fits':0})
    print({'protocol':p['sha256'],'attempts':276,'new_model_calls':0})


def validate():
    p=checked(PROTOCOL);reg,variants=inputs()
    assert p['cells']==matrix(reg) and len(p['cells'])==len({r['id'] for r in p['cells']})==276
    assert p['registry_sha256']==reg['sha256'] and p['limits']==LIMITS
    assert p['ports']==[8004] and p['gpu_indices']==[3] and p['seeds']==SEEDS
    assert p['gpu_memory_utilization']==.35
    for name,h in p['protected_files'].items():assert file_hash(Path(name))==h,name
    assert provenance()['source_hashes']==p['source_hashes']
    assert Counter(c['condition'] for c in p['cells'])=={c:92 for c in CONDITIONS}
    assert {(c['task_id'],c['seed'],c['condition']) for c in p['cells']}=={(r['task_id'],s,c) for r in reg['entries'] for s in SEEDS for c in CONDITIONS}
    return p,variants


def release_owned():
    import subprocess
    path=PRIVATE/'service-ownership.json'
    if not path.exists():return False
    r=read_json(path)
    assert r['gpu']==3 and r['port']==8004 and r['name']=='theory-content-replica-3-20261028'
    current=subprocess.run(['tmux','display-message','-p','-t',r['name'],'#{session_id}|#{session_created}'],capture_output=True,text=True)
    if current.returncode==0 and current.stdout.strip()==r['session_id']+'|'+r['session_created']:
        subprocess.run(['tmux','kill-session','-t',r['name']],check=True);return True
    return False


def run():
    p,variants=validate()
    with run_lock(PRIVATE):
        assert not (PRIVATE/'contract.json').exists(),'Preserve previous dispatches; do not restart'
        before=ledger();assert sum(v['dispatched'] for v in before.values())==sum(v['responses'] for v in before.values())==200396
        freeze=(PRIVATE/'freeze-commit.txt').read_text().strip();assert re_full_sha(freeze)
        transport=CompletionTransport('http://127.0.0.1:8004/v1','local-development',180);health=transport._request('/models')
        assert health['serving_profile']=='reflexion_raw_serial_profiles_v2' and health['dtype']=='bfloat16'
        assert health['service_sha256']==file_hash(Path('scripts/deployment/serve_reflexion_profiles.py'))
        assert health['model_revision']=='b968826d9c46dd6066d109eabc6255188de91218' and health['gpu_memory_utilization']==p['gpu_memory_utilization']
        assert health['model_protocol_sha256']==file_hash(Path('configs/reflexion_qwen3_8b_protocol.json'))
        now=time.time();current=provenance()
        contract=save(PRIVATE/'contract.json',{'protocol_sha256':p['sha256'],'created_unix':now,'deadline_unix':now+43200,
            'max_calls':25000,'prior_ledger':before,'source_hashes':current['source_hashes'],'native_hashes':current['native_hashes'],
            'serving':health,'freeze_commit':freeze,'authorization':'Author approved next intervention on2026-10-28; idle GPU3 only'})
        write_json(PRIVATE/'run.json',{'identity':{'protocol_sha256':p['sha256'],'campaign_sha256':contract['sha256'],
            'source_hashes':current['source_hashes'],'serving':health,'input_format':p['input_format'],
            'decoding_profile':p['decoding_profile'],'environment_action_alias':ACTION_ALIAS_PROFILE,
            'upstream':author_namespace()[1],'wrapper_sha256':file_hash(Path('configs/qwen3_8b_non_thinking_wrapper.json')),
            'model_protocol_sha256':file_hash(Path('configs/reflexion_qwen3_8b_protocol.json'))},'created_unix':now})
        rows=[];status='complete'
        try:
            replica_audit(PRIVATE,p,contract)
            for cell in p['cells']:
                case=p['cases'][cell['index']];memory=variants[cell['index']]['memories'][cell['condition']]
                assert digest(memory)==cell['memory_sha256']
                path=PRIVATE/'cells'/cell['id']/'trial-02'
                r=run_attempt(case['task'],memory,2,path,transport,Path('data/alfworld'),cell['seed'],lambda:reserve_call(PRIVATE,contract),
                    input_format=p['input_format'],decoding_profile=p['decoding_profile'])
                assert r['initial_observation_sha256']==case['initial_observation_sha256'] and r['memory_before']==memory
                assert len(r['memory_after'])==len(memory)+(not r['success'])
                rows.append({**cell,'success':r['success'],'decisions':r['decisions'],
                    'result_path':str(path/'result.json'),'result_file_sha256':file_hash(path/'result.json')})
                save(PRIVATE/'summary.json',{'protocol_sha256':p['sha256'],'status':'running','rows':rows,'attempts_rows':rows,
                    'completed_attempts':len(rows),'expected_attempts':276,'completed_tasks':len(rows)//6,'expected_tasks':46,'elapsed_seconds':time.time()-now})
                print({'completed':len(rows),'total':276,'cell':cell['id'],'success':r['success']},flush=True)
        except CalibrationStop as exc:
            status='allowance_stop';(PRIVATE/'STOP').write_text(str(exc));raise
        except BaseException:
            status='integrity_or_infrastructure_stop';(PRIVATE/'STOP').write_text('Inspect saved evidence before retry');raise
        finally:
            save(PRIVATE/'summary.json',{'protocol_sha256':p['sha256'],'status':status,'rows':rows,'attempts_rows':rows,
                'completed_attempts':len(rows),'expected_attempts':276,'completed_tasks':len(rows)//6,'expected_tasks':46,'elapsed_seconds':time.time()-now})
    assert status=='complete' and len(rows)==276


def close(code):
    try:
        if (PRIVATE/'contract.json').exists():
            p=checked(PROTOCOL);c=checked(PRIVATE/'contract.json');actual=costs(PRIVATE);total=ledger()
            completed=code==0
            if completed:
                result=checked(PUBLIC/'content-intervention-results-20261028.json')
                assert result['checks_passed'] and result['attempts']==276
                assert read_json(PRIVATE/'protocol-audit.json')['passed']
                assert actual['dispatched_calls']==actual['saved_responses']==read_json(PRIVATE/'budget.json')['reserved']
                assert actual['uncertain_or_unrecorded_attempts']==0
            save(PRIVATE/'allowance-closeout.json',{'status':'complete' if completed else 'stopped','allowance_status':'closed',
                'protocol_sha256':p['sha256'],'contract_sha256':c['sha256'],'master_exit_code':code,'costs':actual,
                'global_calls':sum(v['dispatched'] for v in total.values()),'global_responses':sum(v['responses'] for v in total.values()),
                'holdout134_calls':0,'parameter_fits':0})
    finally:print({'released_owned_service':release_owned()},flush=True)


def re_full_sha(s):return len(s)==40 and all(c in '0123456789abcdef' for c in s)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','validate','run','close']);parser.add_argument('--exit-code',type=int,default=1);a=parser.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='validate':p,_=validate();print({'checked_cells':len(p['cells']),'new_model_calls':0})
    elif a.action=='run':run()
    else:close(a.exit_code)
