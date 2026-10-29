#!/usr/bin/env python3
"""One-time native memory interventions and a shared-prefix retention contrast."""
from pathlib import Path
from collections import Counter
import argparse,time,subprocess
from agent_reliability.io import digest,file_hash,read_json,write_json,run_lock
from agent_reliability.cli import provenance
from agent_reliability.reflexion import author_namespace,run_attempt
from agent_reliability.reference import CompletionTransport,ACTION_ALIAS_PROFILE
from prepare_shared_calibration import save
from prepare_four_attempt_protocol import checked
from prepare_recovery_variants import ROOT,PUBLIC,REGISTRY,CONDITIONS,SEEDS
from run_middle_context_transport import reserve_call,replica_audit,CalibrationStop
from run_multiround_diagnostic import verify_attempt
from summarize_reflexion import costs

PROTOCOL=PUBLIC/'recovery-mechanism-protocol-20261029.json'
LIMITS={'additional_calls_including_replica_checks':120000,'seconds':172800}
SOURCE=Path('scripts/run_recovery_mechanism.py')


def matrix(cases):
    out=[]
    for c in cases:
        for j,seed in enumerate(SEEDS):
            shift=(c['index']+j)%5;order=CONDITIONS[shift:]+CONDITIONS[:shift]
            for cond in order:
                out.append(dict(id=f'recovery-{c["task_id"]}-{seed}-{cond}',index=c['index'],task_id=c['task_id'],
                    scene_id=c['scene_id'],seed=seed,condition=cond,max_attempts=4,initial_memory_sha256=c['memory_hashes'][cond]))
    return out


def prepare():
    assert not PROTOCOL.exists() and not (ROOT/'contract.json').exists(),'Preserve prepared collection'
    r=checked(REGISTRY);interface=checked(PUBLIC/'recovery-mechanism-interface-20261029.json');assert interface['registry_sha256']==r['sha256'] and interface['checks_passed']
    from evaluate_recovery_mechanism import engineering_checks
    engineering=engineering_checks()
    from calibrate_successor_success import component_kernel,success_components
    import numpy as np
    parameters=checked(PUBLIC/'successor-development-parameters-20261026.json');model=parameters['models']['seven']['model']
    def entries(labels):return [{'type':a,'native':b} for a,b in labels]
    prior=np.asarray(model['task_prior'])*(1-success_components(model,entries([('N',False)]*3)));prior/=prior.sum()
    initial={c:[('N',False),('N',False),(('E' if c in ['original','paraphrase'] else 'C' if c=='correct' else 'U'),True)] for c in CONDITIONS if c!='delete_entry'}
    initial['delete_entry']=[('N',False)]*2;initial['correct_kept']=initial['correct']
    curves={}
    for condition,state in initial.items():
        cache={}
        def recur(state,left,z,step):
            if not left:return 0.
            if condition=='correct_kept' and step==4:state=(('C',True),)+state[-2:]
            key=(state,left,z,step)
            if key not in cache:
                k=component_kernel(model,entries(state))[z]
                cache[key]=float(k[0]+sum(k[j+1]*recur((state+((typ,True),))[-3:],left-1,z,step+1) for j,typ in enumerate(['E','C','B','U','A'])))
            return cache[key]
        curves[condition]=[float(sum(w*recur(tuple(state),b,z,1) for z,w in enumerate(prior))) for b in range(1,5)]
    assert curves['correct']==curves['remove_target'] and curves['original']==curves['paraphrase']
    assert curves['correct_kept'][:3]==curves['correct'][:3]
    forecasts=save(PUBLIC/'recovery-mechanism-forecasts-20261029.json',dict(parameter_file_sha256=parameters['sha256'],
        model=model,post_parent_failure_prior=prior.tolist(),curves=curves,parameter_fits=0,
        scope='Historical frozen680-row seven-feature model; same prior first NNN failure, G shared across latent classes. Reused development tasks, not independent new-task validation. No selected stage candidate.'))
    protected=checked(PUBLIC/'stage-calibration-protocol-20261029.json')['protected_files'].copy()
    for name in ['scripts/prepare_recovery_mechanism.py','scripts/prepare_recovery_variants.py',str(SOURCE),
        'scripts/evaluate_recovery_mechanism.py','scripts/run_recovery_campaign.sh',
        'docs/experiments/recovery-mechanism-execution-20261029.md',str(REGISTRY),
        str(PUBLIC/'recovery-mechanism-parents-20261029.json'),str(PUBLIC/'recovery-mechanism-interface-20261029.json'),
        'scripts/run_multiround_diagnostic.py','scripts/run_middle_context_transport.py',
        'scripts/diagnose_second_attempt.py','scripts/audit_reflexion_protocol.py','scripts/audit_reflexion_action_shapes.py',
        str(PUBLIC/'recovery-mechanism-forecasts-20261029.json')]:
        protected[name]=file_hash(Path(name))
    p=save(PROTOCOL,dict(scope='Prospective recovery-effect mechanism development on reused native-error parents',
        registry_sha256=r['sha256'],interface_sha256=interface['sha256'],forecasts_sha256=forecasts['sha256'],cases=r['entries'],trajectories=matrix(r['entries']),
        tasks=51,scenes=23,base_trajectories=510,derived_retention_trajectories=102,max_new_attempts=2142,
        conditions=CONDITIONS,seeds=SEEDS,first_logical_trial=2,max_attempts=4,
        ports=[8003,8004],gpu_indices=[2,3],gpu_memory_utilization=.35,limits=LIMITS,prior_global_calls=211234,
        input_format='qwen3_non_thinking_actionline',decoding_profile='qwen3_non_thinking_recommended_v1',
        source_hashes=provenance()['source_hashes'],protected_files=protected,
        analysis=dict(primary=[['correct','paraphrase'],['correct','delete_entry']],primary_budget=4,
            aggregation='Two paired seeds mean per task; tasks mean per scene;23 scenes equal',
            interval='Two coprimary contrasts: paired scene t97.5%,df22,Bonferroni family .05',
            t_primary=interface['primary_t97_5_critical_df22'],t_secondary=interface['secondary_t95_critical_df22'],
            secondary='All1-4 curves, other paired differences, correction-retention contrast, actions/costs; no selective claim from nominal intervals',
            stopping='Complete full matrix or stop for declared allowance/integrity; no outcome-based sample-size change',
            write_labels='Save all failed writes; semantic E labels/joint failed-and-E events require postcollection full-text review, not lexical auto-confirmation'),
        retention=dict(condition='correct_kept',parent='correct',
            rule='Share identical first3 corrected attempts. If all3fail, fourth input copies initial corrected entry into oldest of latest3, replacing J1; keep J2,J3. Same logicaltrial5/seed. If success<=3, copy stopped outcome without new call.',
            interpretation='Retaining original correction instead of earliest new reflection, not pure age effect or neutral padding'),
        engineering_checks=engineering,parameter_fits=0,holdout134_calls=0))
    print(dict(protocol_sha256=p['sha256'],trajectories=510,derived=102,max_attempts=2142,limits=LIMITS),flush=True)


def validate():
    p=checked(PROTOCOL);r=checked(REGISTRY);v=read_json(ROOT/'variants.json')
    assert p['registry_sha256']==r['sha256'] and file_hash(ROOT/'variants.json')==r['private_variants_sha256']
    assert p['trajectories']==matrix(p['cases']) and len(p['trajectories'])==510
    assert Counter(t['condition'] for t in p['trajectories'])=={c:102 for c in CONDITIONS}
    assert p['limits']==LIMITS and p['ports']==[8003,8004] and p['gpu_indices']==[2,3]
    for name,h in p['protected_files'].items():assert file_hash(Path(name))==h,name
    assert provenance()['source_hashes']==p['source_hashes']
    return p,v


def release():
    released=[]
    for gpu,filename in [(2,'service-ownership-2.json'),(3,'service-ownership.json')]:
        path=ROOT/filename
        if not path.exists():continue
        r=read_json(path);assert r['gpu']==gpu and r['port']==8001+gpu and r['name']==f'theory-recovery-replica-{gpu}-20261029'
        current=subprocess.run(['tmux','display-message','-p','-t',r['name'],'#{session_id}|#{session_created}'],capture_output=True,text=True)
        if current.returncode==0 and current.stdout.strip()==r['session_id']+'|'+r['session_created']:
            subprocess.run(['tmux','kill-session','-t',r['name']],check=True);released.append(gpu)
    return released


def run():
    p,variants=validate()
    with run_lock(ROOT):
        assert not (ROOT/'contract.json').exists(),'No automatic restart'
        from continue_reflexion_pilot import ledger
        before=ledger();assert sum(a['dispatched'] for a in before.values())==p['prior_global_calls']
        assert all(a['dispatched']==a['responses'] for a in before.values())
        freeze=(ROOT/'freeze-commit.txt').read_text().strip();assert len(freeze)==40
        transport=CompletionTransport('http://127.0.0.1:8003/v1','local-development',180);health=transport._request('/models')
        assert CompletionTransport('http://127.0.0.1:8004/v1','local-development',180)._request('/models')==health
        assert health['serving_profile']=='reflexion_raw_serial_profiles_v2' and health['dtype']=='bfloat16'
        assert health['model_revision']=='b968826d9c46dd6066d109eabc6255188de91218' and health['gpu_memory_utilization']==.35
        assert health['service_sha256']==file_hash(Path('scripts/deployment/serve_reflexion_profiles.py'))
        assert health['model_protocol_sha256']==file_hash(Path('configs/reflexion_qwen3_8b_protocol.json'))
        own=min([read_json(ROOT/n) for n in ['service-ownership.json','service-ownership-2.json']],key=lambda r:r['created_unix']); now=time.time(); current=provenance()
        assert 0<=now-own['created_unix']<LIMITS['seconds']
        contract=save(ROOT/'contract.json',dict(protocol_sha256=p['sha256'],freeze_commit=freeze,created_unix=now,
            deadline_unix=own['created_unix']+LIMITS['seconds'],max_calls=LIMITS['additional_calls_including_replica_checks'],
            prior_ledger=before,serving=health,source_hashes=current['source_hashes'],native_hashes=current['native_hashes'],
            authorization='User authorized continuation of recovery-centered plan,2026-10-29; preserve other GPU jobs'))
        write_json(ROOT/'run.json',dict(created_unix=now,identity=dict(protocol_sha256=p['sha256'],campaign_sha256=contract['sha256'],
            serving=health,source_hashes=current['source_hashes'],upstream=author_namespace()[1],input_format=p['input_format'],
            decoding_profile=p['decoding_profile'],environment_action_alias=ACTION_ALIAS_PROFILE,
            wrapper_sha256=file_hash(Path('configs/qwen3_8b_non_thinking_wrapper.json')),
            model_protocol_sha256=file_hash(Path('configs/reflexion_qwen3_8b_protocol.json')))))
        def worker(index):
            transport=CompletionTransport(f'http://127.0.0.1:{p["ports"][index]}/v1','local-development',180)
            allrows=[];completed=[];derived=[];status='running'
            def store():
                save(ROOT/f'worker-{index}.json',dict(protocol_sha256=p['sha256'],status=status,attempts_rows=allrows,
                    trajectories=completed,retention_trajectories=derived,expected_trajectories=510,completed_trajectories=len(completed),
                    derived_completed=len(derived),expected_derived=102,attempts_completed=len(allrows),
                    completed_tasks=len(completed)//10,expected_tasks=51,elapsed_seconds=time.time()-now))
            def one(case,tr,memory,step,condition):
                trial=step+1;path=ROOT/'trajectories'/tr['id']/f'trial-{trial:02d}'
                result=run_attempt(case['task'],list(memory),trial,path,transport,Path('data/alfworld'),tr['seed'],lambda:reserve_call(ROOT,contract),
                    input_format=p['input_format'],decoding_profile=p['decoding_profile'])
                verify_attempt(result,path,case,tr,memory,trial)
                row=dict(trajectory_id=tr['id'],task_id=case['task_id'],scene_id=case['scene_id'],seed=tr['seed'],
                    condition=condition,trial=trial,attempt_index=step,success=result['success'],decisions=result['decisions'],
                    memory_before_sha256=digest(memory),visible_memory_sha256=digest(memory[-3:]),
                    new_reflection_sha256=[digest(a) for a in result['memory_after'][len(memory):]],
                    result_path=str(path/'result.json'),result_file_sha256=file_hash(path/'result.json'))
                allrows.append(row);store();print(dict(attempts=len(allrows),completed=len(completed),id=tr['id'],step=step,success=result['success']),flush=True)
                return result,row
            try:
                store()
                for tr in p['trajectories']:
                    if tr['index']%2!=index:continue
                    case=p['cases'][tr['index']];initial=variants[tr['index']]['memories'][tr['condition']];memory=list(initial);rows=[];snapshots=[]
                    assert digest(initial)==tr['initial_memory_sha256']
                    for step in range(1,5):
                        snapshots.append(list(memory));result,row=one(case,tr,memory,step,tr['condition']);rows.append(row);memory=result['memory_after']
                        if result['success']:break
                    completed.append(dict(tr,complete=True,success=any(r['success'] for r in rows),attempts=rows))
                    if tr['condition']=='correct':
                        dt={**tr,'id':tr['id']+'_kept','condition':'correct_kept','shared_parent_trajectory':tr['id']}
                        if any(r['success'] for r in rows[:3]):
                            derived.append(dict(dt,complete=True,shared_prefix=rows,alternative=None,success=True))
                        else:
                            assert len(rows)==4
                            mem=list(snapshots[3]);assert len(mem)==6 and mem[-3:]==memory_before_last(rows[3])[-3:]
                            mem[-3]=initial[-1];result,alt=one(case,dt,mem,4,'correct_kept')
                            derived.append(dict(dt,complete=True,shared_prefix=rows[:3],alternative=alt,success=result['success']))
                    store()
                assert len(completed)==(260 if index==0 else 250) and len(derived)==(52 if index==0 else 50)
                status='complete';store()
            except CalibrationStop as exc:
                status='allowance_stop';(ROOT/'STOP').write_text(str(exc));raise
            except BaseException:
                status='integrity_or_infrastructure_stop';(ROOT/'STOP').write_text('Inspect before retry; no automatic restart');raise
            finally:store()
        replica_audit(ROOT,p,contract)
        import multiprocessing
        processes=[multiprocessing.get_context('fork').Process(target=worker,args=(i,)) for i in range(2)]
        for child in processes:child.start()
        def merge(status):
            workers=[read_json(ROOT/f'worker-{i}.json') for i in range(2) if (ROOT/f'worker-{i}.json').exists()]
            rows=[r for w in workers for r in w['attempts_rows']];main=[r for w in workers for r in w['trajectories']];derived=[r for w in workers for r in w['retention_trajectories']]
            counts=Counter(r['task_id'] for r in main)
            return save(ROOT/'summary.json',dict(protocol_sha256=p['sha256'],status=status,
                attempts_rows=sorted(rows,key=lambda r:(r['trajectory_id'],r['attempt_index'])),trajectories=main,retention_trajectories=derived,
                expected_trajectories=510,completed_trajectories=len(main),derived_completed=len(derived),expected_derived=102,
                attempts_completed=len(rows),completed_tasks=sum(v==10 for v in counts.values()),expected_tasks=51,
                worker_statuses=[w['status'] for w in workers],elapsed_seconds=time.time()-now))
        while any(child.is_alive() for child in processes):
            merge('running');time.sleep(10)
        for child in processes:child.join()
        report=merge('checking')
        complete=all(child.exitcode==0 for child in processes) and report['completed_trajectories']==510 and report['derived_completed']==102 and report['worker_statuses']==['complete','complete']
        merge('complete' if complete else 'partial')
        assert complete,'Incomplete campaign; inspect STOP and workers before any retry'


def memory_before_last(row):return read_json(Path(row['result_path']))['memory_before']


def close(code):
    try:
        if (ROOT/'contract.json').exists():
            from continue_reflexion_pilot import ledger
            p=checked(PROTOCOL);c=checked(ROOT/'contract.json');a=costs(ROOT);complete=code==0
            if complete:
                result=checked(PUBLIC/'recovery-mechanism-results-20261029.json');assert result['checks_passed']
                assert read_json(ROOT/'protocol-audit.json')['passed']
                assert a['dispatched_calls']==a['saved_responses']==read_json(ROOT/'budget.json')['reserved']
                assert a['uncertain_or_unrecorded_attempts']==0
            allcalls=ledger()
            save(ROOT/'allowance-closeout.json',dict(status='complete' if complete else 'stopped',allowance_status='closed',
                protocol_sha256=p['sha256'],contract_sha256=c['sha256'],master_exit_code=code,costs=a,
                global_calls=sum(a['dispatched'] for a in allcalls.values()),global_responses=sum(a['responses'] for a in allcalls.values()),
                parameter_fits=0,holdout134_calls=0))
    finally:print(dict(released_owned_service=release()),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','validate','run','close']);p.add_argument('--exit-code',type=int,default=1);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='validate':validate();print('recovery matrix/source checks passed;zero calls')
    elif a.action=='run':run()
    else:close(a.exit_code)
