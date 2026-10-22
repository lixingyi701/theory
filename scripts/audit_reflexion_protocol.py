#!/usr/bin/env python3
"""Audit the recorded model/action boundary; no generation or semantic repairs."""
import argparse
import re
from pathlib import Path

from agent_reliability.io import digest, file_hash, read_json, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    protocol_path = Path('configs/reflexion_qwen3_8b_protocol.json')
    contract = read_json(protocol_path)
    run = read_json(a.run/'run.json')['identity']
    profile=run.get('decoding_profile','author_reflexion')
    wrapper = read_json(Path(contract['wrapper_file']))
    errors = []
    for field in ['input_format', 'environment_action_alias']:
        if run[field] != contract[field]:
            errors.append({'rule': field, 'path': 'run.json'})
    if run['serving']['model_revision'] != contract['model_revision'] or run['upstream']['commit'] != contract['upstream_reflexion_commit']:
        errors.append({'rule': 'model_or_author_revision', 'path': 'run.json'})
    if run['wrapper_sha256'] != file_hash(Path(contract['wrapper_file'])):
        errors.append({'rule': 'wrapper_identity', 'path': 'run.json'})
    if run.get('model_protocol_sha256',file_hash(protocol_path))!=file_hash(protocol_path):
        errors.append({'rule':'protocol_identity','path':'run.json'})
    if run['serving'].get('model_protocol_sha256',file_hash(protocol_path))!=file_hash(protocol_path):
        errors.append({'rule':'server_protocol_identity','path':'run.json'})
    totals = {'dispatched_calls': 0, 'responses': 0, 'missing_responses': 0,
        'leading_action_markers': 0, 'empty_responses': 0, 'truncated_responses': 0,
        'placement_aliases': 0, 'in_on_variants': 0, 'no_effect_steps': 0, 'environment_steps': 0}
    totals.update(thought_steps=0,physical_no_effect_steps=0,unexpected_thinking_blocks=0)
    for request in sorted(a.run.glob('**/calls/*/attempt-*.request.json')):
        totals['dispatched_calls'] += 1
        saved = read_json(request); body = saved['payload']; role = request.parent.name.split('-')[0]
        expected = dict(contract['actor' if role == 'actor' else 'reflector'])
        if profile=='qwen3_non_thinking_recommended_v1':
            expected.update(contract['official_non_thinking_recommendation']);expected.pop('temperatures',None)
        bad = []
        if saved['payload_sha256'] != digest(body): bad.append('request_hash')
        prompt = body['prompt']
        if not prompt.startswith(wrapper['prefix']) or not prompt.endswith(wrapper['suffix']) or prompt.count(wrapper['prefix']) != 1:
            bad.append('single_official_wrapper')
        if body['max_tokens'] != expected['max_tokens'] or body['top_p'] != expected['top_p'] or body['stop'] != expected['stop']:
            bad.append('decoding_contract')
        temperatures = expected.get('temperatures', [expected.get('temperature')])
        if not any(abs(body['temperature']-x) < 1e-8 for x in temperatures): bad.append('role_temperature')
        if profile=='qwen3_non_thinking_recommended_v1':
            if body.get('sampling_profile')!=profile or body.get('top_k')!=20 or body.get('min_p')!=0 or body.get('stop_token_ids')!=contract['model_eos_token_ids']:
                bad.append('official_sampling_profile')
        if type(body['seed']) is not int: bad.append('explicit_seed')
        response_path = request.parent/'response.json'
        if not response_path.exists():
            totals['missing_responses'] += 1
        else:
            response = read_json(response_path)
            totals['responses'] += 1
            if response['payload_sha256'] != saved['payload_sha256']: bad.append('response_identity')
            if response['response'].get('serving_profile')!=run['serving']['serving_profile']: bad.append('response_serving_profile')
            if response['response'].get('sampling_profile',profile)!=profile: bad.append('response_sampling_profile')
            choice = response['response']['choices'][0]
            text = choice['text']
            if '<think>' in text or '</think>' in text:
                totals['unexpected_thinking_blocks']+=1;bad.append('non_thinking_mode')
            totals['empty_responses'] += not bool(text.strip())
            totals['truncated_responses'] += choice.get('finish_reason') == 'length'
            totals['leading_action_markers'] += role == 'actor' and text.lstrip().startswith('> ')
            usage = response['response'].get('usage')
            if not usage or usage['total_tokens'] > contract['max_context_tokens']: bad.append('context_or_usage')
            effective=response['response'].get('effective_sampling_params')
            if profile=='qwen3_non_thinking_recommended_v1' and not effective: bad.append('resolved_sampling_missing')
            if effective:
                for field in ['temperature','top_p','max_tokens','seed','frequency_penalty','presence_penalty']:
                    if effective[field]!=body[field]: bad.append('resolved_'+field)
                if effective['stop']!=(body['stop'] or []): bad.append('resolved_stop')
                if effective['top_k']!=body.get('top_k',-1) or effective['min_p']!=body.get('min_p',0): bad.append('resolved_sampling_filters')
                if effective['ignore_eos'] is not False: bad.append('resolved_eos_handling')
                if 'stop_token_ids' in body and effective.get('stop_token_ids')!=body['stop_token_ids']: bad.append('resolved_eos_ids')
        errors.extend({'rule': rule, 'path': str(request.relative_to(a.run))} for rule in bad)
    for path in sorted(a.run.glob('**/checkpoint.json')):
        for step in read_json(path)['transitions']:
            totals['environment_steps'] += 1
            raw = step['action']; match = re.fullmatch(r'put (.+) (in/on|in|on) (.+)', raw)
            expected = f'move {match[1]} to {match[3]}' if match else raw
            if expected != step['submitted_action']:
                errors.append({'rule': 'entity_preserving_tool_translation', 'path': str(path.relative_to(a.run))})
            totals['placement_aliases'] += bool(match)
            totals['in_on_variants'] += bool(match and match[2] in {'in', 'on'})
            totals['no_effect_steps'] += step['observation'] == 'Nothing happens.'
            totals['thought_steps']+=raw.startswith('think:')
            totals['physical_no_effect_steps']+=not raw.startswith('think:') and step['observation']=='Nothing happens.'
    if totals['missing_responses']:
        errors.append({'rule': 'unconfirmed_requests', 'path': 'calls/'})
    summary = read_json(a.run/'summary.json')
    if summary['completed_tasks'] != summary['expected_tasks']:
        errors.append({'rule': 'incomplete_task_cohort', 'path': 'summary.json'})
    report = {'scope': 'model and tool interface audit; no-effect observations are not automatically classified as syntax errors',
        'decoding_profile':profile,
        'protocol_sha256': file_hash(protocol_path), 'run_sha256': file_hash(a.run/'run.json'),
        'driver_sha256': file_hash(Path(__file__)), 'passed': not errors, 'totals': totals, 'errors': errors}
    report['sha256'] = digest(report); write_json(a.output, report)
    print({'passed': report['passed'], 'totals': totals, 'errors': len(errors)})
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
