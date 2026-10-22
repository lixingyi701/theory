"""Experiment orchestration for unchanged author ReAct functions, not an agent."""
from collections import Counter
import os
from pathlib import Path
from .audit import verify_audit
from .dataset import load_manifest
from .io import digest, file_hash, read_json, write_json
from .reference import CompletionTransport, author_code, official_environment, run_reference_episode
from .llm import CallJournal


def summarize_reference(directory):
    run=read_json(directory/'run.json');rows=[read_json(p) for p in directory.glob('episodes/*/result.json')]
    n=run['expected_episodes']
    journals=[CallJournal(p,None,49).totals() for p in directory.glob('episodes/*/calls')]
    report={'scope':'author ReAct control flow with 8B substitution; development only, not paper score or theory evidence',
            'expected_episodes':n,'completed_episodes':len(rows),'missing_episodes':n-len(rows),'complete':n==len(rows),
            'successes':sum(x['success'] for x in rows),'status_counts':dict(Counter(x['status'] for x in rows)),
            'successful_types':sorted({x['task_type'] for x in rows if x['success']}),
            'by_type':{kind:{'completed':sum(x['task_type']==kind for x in rows),
                             'successes':sum(x['success'] for x in rows if x['task_type']==kind)}
                       for kind in sorted({x['task_type'] for x in run['tasks']})},
            'costs':{k:sum(x[k] for x in journals) for k in ['dispatched_calls','saved_responses',
                 'input_tokens_known','output_tokens_known','truncated_responses','uncertain_or_unrecorded_attempts']}}
    write_json(directory/'summary.json',report)
    return report


def run_reference_command(args):
    from .cli import provenance
    config=read_json(args.config);manifest=load_manifest(args.manifest);model=read_json(args.models)[config['model_key']]
    if config['method']!='author_react_alfworld' or config['model_key']!='qwen3_8b' or config['conditions']!=['author_react_base']:
        raise ValueError('This entry is the author ReAct 8B baseline; no custom memory/recovery')
    if model['revision']!='b968826d9c46dd6066d109eabc6255188de91218':
        raise ValueError('Pinned 8B revision required')
    _,upstream=author_code(Path(config['upstream_dir']))
    audit=read_json(args.audit_report);verify_audit(audit,manifest)
    env_audit=read_json(args.environment_report)
    if digest({k:v for k,v in env_audit.items() if k!='sha256'}) != env_audit['sha256']:
        raise ValueError('Official environment report integrity mismatch')
    if (not env_audit['passed'] or env_audit['restore_checks_passed']<40 or len(env_audit['tasks'])<10
        or env_audit['manifest_sha256']!=manifest['sha256'] or env_audit['direct_audit_sha256']!=audit['sha256']
        or env_audit['upstream']!=upstream):
        raise ValueError('Require matched official-environment 10-task/40-check equivalence audit')
    current=provenance()
    if (audit.get('provenance',{}).get('native_hashes')!=current['native_hashes']
        or env_audit.get('native_hashes')!=current['native_hashes']):
        raise ValueError('Planner binary changed after audit; rerun both environment audits')
    from .alfworld import ALFWORLD_COMMIT
    alf=current['packages'].get('alfworld') or {}
    if ((alf.get('direct_url') or {}).get('vcs_info') or {}).get('commit_id')!=ALFWORLD_COMMIT:
        raise ValueError('ALFWorld pinned source installation required')
    for name in ['alfworld.py','env.py','audit.py','reference.py']:
        if env_audit['source_hashes'][name]!=current['source_hashes'][name]:
            raise ValueError('Backend/adapter changed after official-environment audit')
    transport=CompletionTransport(os.environ.get(config['api_base_env'],'http://127.0.0.1:8000/v1'),
                                  os.environ.get(config['api_key_env'],'local-development'),config['timeout_seconds'])
    health=transport._request('/models')
    if (health.get('serving_profile')!=config['serving_profile'] or health.get('model_revision')!=model['revision']
        or 'raw_completions' not in health.get('capabilities',[])):
        raise ValueError('Wrong serving identity/capability; no experiment request sent')
    identity={'manifest_sha256':manifest['sha256'],'config':config,'model':model,'upstream':upstream,
              'audit_sha256':audit['sha256'],'environment_audit_sha256':env_audit['sha256'],
              'source_hashes':current['source_hashes'],'serving_identity':health}
    if (args.output/'run.json').exists():
        if not args.resume or read_json(args.output/'run.json')['identity_sha256']!=digest(identity):
            raise ValueError('Run identity changed; use a new directory')
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError('Output not empty')
        write_json(args.output/'run.json',{**identity,'identity_sha256':digest(identity),'provenance':current,
                   'tasks':manifest['tasks'],'expected_episodes':len(manifest['tasks'])*len(config['seeds'])})
    for task in manifest['tasks']:
        for seed in config['seeds']:
            env=official_environment(args.data_root,task,seed,Path(config['upstream_dir']))
            try:
                row=run_reference_episode(env,transport,task,config,model,
                    args.output/'episodes'/f"{task['task_id']}-s{seed}",seed,args.resume,args.retry_uncertain)
            finally:
                env.close()
            print({k:row[k] for k in ['task_id','success','status','decisions']},flush=True)
            report=summarize_reference(args.output)
            if args.stop_after_episodes and report['completed_episodes']>=args.stop_after_episodes:
                return report
            if report['completed_episodes']>=config['early_stop_zero_successes_after'] and not report['successes']:
                return report
    return summarize_reference(args.output)
