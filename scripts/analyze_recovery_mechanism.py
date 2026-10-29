#!/usr/bin/env python3
"""Postcollection independent audit. No inference, fitting or frozen-source edits."""
import collections as co
import hashlib
import json
import math
from pathlib import Path
import re
import statistics as st
import subprocess
import time

ROOT = Path('runs/reflexion-recovery-mechanism-20261029')
PUBLIC = Path('repro/reflexion-20261025')
CONDITIONS = ['original', 'paraphrase', 'correct', 'remove_target', 'delete_entry', 'correct_kept']


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def checked(p):
    x = read(p)
    assert x['sha256'] == digest({k: v for k, v in x.items() if k != 'sha256'}), str(p)
    return x


def save(p, x):
    x = dict(x, sha256=digest(x))
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(x, ensure_ascii=False, sort_keys=True, indent=2,
                                 allow_nan=False) + '\n', encoding='utf-8')
    return x


def seed_for(*x):
    return int(digest(x)[:8], 16) % (2**31 - 1)


def parse_trace(cp, target):
    held, placed = set(), set()
    pickup = heat = bad = physical = no_effect = thought = 0
    for t in cp['transitions']:
        a, obs = t.get('submitted_action', t['action']), t['observation']
        if t['action'].lower().startswith('think:'):
            thought += 1
            continue
        physical += 1
        no_effect += obs.strip() == 'Nothing happens.'
        m = re.search(r'(?:^|:\s*)You pick up the (' + re.escape(target) + r' \d+)(?: from |\.|$)', obs, re.I)
        if m:
            held.add(m[1].lower())
            placed.discard(m[1].lower())
            pickup += 1
        m = re.search(r'(?:^|:\s*)You (?:put|move) the (' + re.escape(target) + r' \d+) (?:in|on|into|to) (?:the )?(.+?)\.?$', obs, re.I)
        if m:
            held.discard(m[1].lower())
            if m[2].lower().startswith('microwave '):
                placed.add(m[1].lower())
        m = re.fullmatch(r'heat (' + re.escape(target) + r' \d+) with (.+)', a, re.I)
        if m and m[1].lower() in placed and m[1].lower() not in held:
            bad += 1
        heat += bool(re.match(r'^You heat the ' + re.escape(target) + r' \d+ using ', obs, re.I))
    won = bool(cp['transitions'][-1]['won'])
    stage = 'won' if won else 'no_target_pickup' if not pickup else 'pickup_without_heat' if not heat else 'heated_without_win'
    return dict(heated=bool(heat), put_then_heat=bool(bad), bad_order_steps=bad,
                target_heat_steps=heat, physical_steps=physical, thought_steps=thought,
                physical_no_effect=no_effect, failure_stage=stage)


def scene_values(rows, c, b, field='success_curve'):
    tasks = co.defaultdict(lambda: co.defaultdict(list))
    for r in rows:
        if r['condition'] == c:
            value = r[field][b-1] if isinstance(r[field], list) else r[field]
            tasks[r['scene_id']][r['task_id']].append(float(value))
    assert len(tasks) == 23
    assert all(len(v) == 2 for d in tasks.values() for v in d.values())
    return {s: st.mean(st.mean(v) for v in d.values()) for s, d in sorted(tasks.items())}


def contrast(rows, left, right, b, critical, field='success_curve'):
    a, z = scene_values(rows, left, b, field), scene_values(rows, right, b, field)
    ds = {s: a[s]-z[s] for s in a}
    m, se = st.mean(ds.values()), st.stdev(ds.values()) / math.sqrt(23)
    return dict(left=left, right=right, budget=b, field=field, difference=m,
                paired_scene_se=se, interval=[m-critical*se, m+critical*se], scene_differences=ds)


def main():
    start = time.time()
    assert not (ROOT/'independent-analysis.json').exists(), 'Preserve prior audit; investigate before repeating'
    p = checked(PUBLIC/'recovery-mechanism-protocol-20261029.json')
    assert p['sha256'] == 'ad164d31101a62246469b3c5dd11714937d9783d797db4508af203c376e86d31'
    assert len(p['protected_files']) == 95
    for f, h in p['protected_files'].items():
        assert sha(f) == h, f
    summary = checked(ROOT/'summary.json')
    close = checked(ROOT/'allowance-closeout.json')
    scored = checked(PUBLIC/'recovery-mechanism-results-20261029.json')
    observed = checked(ROOT/'observed.json')
    assert summary['status'] == close['status'] == 'complete'
    assert close['allowance_status'] == 'closed' and close['master_exit_code'] == 0
    assert not (ROOT/'STOP').exists()
    assert 'RECOVERY_MECHANISM_AND_EVALUATION_COMPLETE' in (ROOT/'master.log').read_text()
    audit = checked(ROOT/'protocol-audit.json')
    assert audit['passed'] and audit['totals']['missing_responses'] == 0
    assert summary['completed_trajectories'] == 510 and summary['derived_completed'] == 102
    cases = {r['task_id']: r for r in p['cases']}
    variants = read(ROOT/'variants.json')
    prompts = {(r['index'], r['condition']): r['wrapped_prompt'] for r in read(ROOT/'interface-prompts.json')}
    actual, traces = {}, {}
    projected = []
    for row in summary['attempts_rows']:
        path = Path(row['result_path'])
        result, cp, identity = read(path), read(path.with_name('checkpoint.json')), checked(path.with_name('identity.json'))
        case = cases[row['task_id']]
        assert sha(path) == row['result_file_sha256']
        assert bool(cp['transitions'][-1]['won']) == result['success'] == row['success']
        assert all(not t['won'] for t in cp['transitions'][:-1])
        assert result['identity_sha256'] == identity['sha256']
        assert identity['task'] == case['task'] and identity['seed'] == row['seed']
        assert row['seed'] in [2190, 2191]
        assert result['trial'] == identity['trial'] == row['trial'] == row['attempt_index']+1
        assert identity['memory'] == result['memory_before']
        assert result['initial_observation_sha256'] == case['initial_observation_sha256']
        assert identity['decoding_config_sha256'] == audit['protocol_sha256']
        assert identity['decoding_profile'] == 'qwen3_non_thinking_recommended_v1'
        assert 0 < result['decisions'] <= 49
        before, after = result['memory_before'], result['memory_after']
        assert after[:len(before)] == before
        assert after == before if result['success'] else len(after) == len(before)+1 and bool(after[-1].strip())
        assert digest(before) == row['memory_before_sha256']
        assert digest(before[-3:]) == row['visible_memory_sha256']
        assert row['new_reflection_sha256'] == ([] if result['success'] else [digest(after[-1])])
        req = read(path.parent/'calls/actor-0000/attempt-000.request.json')['payload']
        assert req['seed'] == seed_for(row['task_id'], row['seed'], row['trial'], 0, 'actor')
        if row['attempt_index'] == 1:
            assert req['prompt'] == prompts[case['index'], row['condition']]
        key = (row['trajectory_id'], row['attempt_index'])
        assert key not in actual
        actual[key] = result
        target = Path(case['task']['gamefile']).parts[-3].split('-')[1].lower()
        parsed = parse_trace(cp, target)
        traces[key] = parsed
        projected.append(dict(row, checkpoint_file_sha256=sha(path.with_name('checkpoint.json')),
                              identity_file_sha256=sha(path.with_name('identity.json')), **parsed,
                              dispatched_calls=result['dispatched_calls'], decisions=result['decisions'],
                              truncated_responses=result['truncated_responses']))
    assert len(actual) == summary['attempts_completed'] == 1594
    records = []
    bases = {tr['id']: tr for tr in summary['trajectories']}
    def make_record(tr, rr, condition):
        prefix = [actual[x['trajectory_id'], x['attempt_index']] for x in rr]
        ts = [traces[x['trajectory_id'], x['attempt_index']] for x in rr]
        return dict(trajectory_id=tr['id'], task_id=tr['task_id'], scene_id=tr['scene_id'], seed=tr['seed'], condition=condition,
                    success_curve=[any(x['success'] for x in prefix[:b]) for b in range(1, 5)],
                    heated=any(x['heated'] for x in ts), put_then_heat=any(x['put_then_heat'] for x in ts),
                    attempts=len(rr), calls=sum(x['dispatched_calls'] for x in prefix),
                    truncated_responses=sum(x['truncated_responses'] for x in prefix),
                    physical_no_effect=sum(x['physical_no_effect'] for x in ts))
    for tr in bases.values():
        memory = variants[tr['index']]['memories'][tr['condition']]
        assert digest(memory) == tr['initial_memory_sha256']
        for j, row in enumerate(tr['attempts'], 1):
            result = actual[tr['id'], j]
            assert result['memory_before'] == memory
            assert result['success'] == (j == len(tr['attempts']) and tr['success'])
            memory = result['memory_after']
        assert tr['complete'] and (tr['success'] or len(tr['attempts']) == 4)
        records.append(make_record(tr, tr['attempts'], tr['condition']))
    extra = 0
    retention_pairs = co.Counter()
    for tr in summary['retention_trajectories']:
        parent = bases[tr['shared_parent_trajectory']]
        assert parent['condition'] == 'correct'
        assert tr['shared_prefix'] == parent['attempts'][:3]
        assert (tr['task_id'], tr['scene_id'], tr['seed']) == (parent['task_id'], parent['scene_id'], parent['seed'])
        rr = list(tr['shared_prefix'])
        if tr['alternative'] is None:
            assert any(x['success'] for x in rr)
        else:
            extra += 1
            assert not any(x['success'] for x in rr) and len(rr) == 3
            alt = actual[tr['id'], 4]
            expect = list(actual[parent['id'], 4]['memory_before'])
            expect[-3] = variants[tr['index']]['memories']['correct'][-1]
            assert alt['memory_before'] == expect
            rr.append(tr['alternative'])
            retention_pairs[str((bool(actual[parent['id'], 4]['success']), bool(alt['success'])))] += 1
        record = make_record(tr, rr, 'correct_kept')
        assert record['success_curve'][:3] == records[next(i for i,x in enumerate(records) if x['trajectory_id'] == parent['id'])]['success_curve'][:3]
        records.append(record)
    assert len(records) == 612 and extra == 49
    assert len({(x['task_id'], x['seed'], x['condition']) for x in records}) == 612
    frozen = checked(PUBLIC/'recovery-mechanism-forecasts-20261029.json')
    curves = {}
    for c in CONDITIONS:
        rows = [r for r in records if r['condition'] == c]
        assert len(rows) == 102
        curve = [st.mean(scene_values(records, c, b).values()) for b in range(1, 5)]
        counts = [sum(r['success_curve'][b] for r in rows) for b in range(4)]
        assert curve == scored['curves'][c]['scene_equal_curve'] and counts == scored['curves'][c]['raw_success_counts']
        curves[c] = dict(scene_equal_curve=curve, raw_success_counts=counts, frozen_forecast=frozen['curves'][c],
                         heated_trajectories=sum(r['heated'] for r in rows), bad_order_trajectories=sum(r['put_then_heat'] for r in rows),
                         strategy_attempts=sum(r['attempts'] for r in rows), strategy_calls=sum(r['calls'] for r in rows),
                         strategy_truncations=sum(r['truncated_responses'] for r in rows))
    contrasts = []
    for ref in scored['contrasts']:
        crit = p['analysis']['t_primary'] if ref['primary'] else p['analysis']['t_secondary']
        new = contrast(records, ref['left'], ref['right'], ref['budget'], crit)
        assert abs(new['difference'] - ref['difference']) < 1e-14
        assert all(abs(a-b) < 1e-14 for a,b in zip(new['interval'], ref['interval']))
        contrasts.append(dict(new, primary=ref['primary'], coverage=ref['coverage'],
                              frozen_predicted_difference=ref['frozen_predicted_difference']))
    q = read(ROOT/'failed-write-label-queue.json')['rows']
    expected = {(r['trajectory_id'], r['attempt_index']) for r in summary['attempts_rows'] if not r['success']}
    assert len(q) == len(expected) == 1349
    for row in q:
        key = (row['trajectory_id'], row['attempt_index'])
        assert key in expected and row['text'] == actual[key]['memory_after'][-1]
        assert digest(row['text']) == row['text_sha256'] and row['label'] is None
    # Every physical response, including the four replica checks, is counted once.
    requests = list(ROOT.rglob('attempt-*.request.json'))
    responses = list(ROOT.rglob('response.json'))
    assert len(requests) == len(responses) == read(ROOT/'budget.json')['reserved'] == 66372
    ledger = co.Counter()
    for path in requests:
        request = read(path)
        response = read(path.with_name('response.json'))
        assert request['payload_sha256'] == digest(request['payload']) == response['payload_sha256']
        assert response['attempt'] == 0 and path.name == 'attempt-000.request.json'
        usage = response['response']['usage']
        ledger['input_tokens'] += usage.get('prompt_tokens', 0)
        ledger['output_tokens'] += usage.get('completion_tokens', 0)
        ledger['truncated_responses'] += any(x.get('finish_reason') == 'length' for x in response['response']['choices'])
    assert ledger['input_tokens'] == close['costs']['input_tokens_known']
    assert ledger['output_tokens'] == close['costs']['output_tokens_known']
    assert ledger['truncated_responses'] == 19
    actual_costs = {k: sum(x[k] for x in actual.values()) for k in ['dispatched_calls', 'saved_responses', 'input_tokens_known',
                    'output_tokens_known', 'truncated_responses', 'usage_missing_responses', 'uncertain_or_unrecorded_attempts']}
    assert actual_costs['dispatched_calls']+4 == len(requests)
    own_names = [read(ROOT/n)['name'] for n in ['service-ownership.json', 'service-ownership-2.json']]
    tmux = subprocess.run(['tmux', 'list-sessions', '-F', '#{session_name}'], capture_output=True, text=True)
    assert all(n not in tmux.stdout.splitlines() for n in own_names)
    action_by_condition = {}
    for c in CONDITIONS:
        rows = [r for r in projected if r['condition'] == c]
        if not rows:
            continue
        action_by_condition[c] = dict(actual_attempts=len(rows),
            **{k: sum(r[k] for r in rows) for k in ['heated', 'put_then_heat', 'bad_order_steps', 'target_heat_steps',
                                                  'physical_steps', 'thought_steps', 'physical_no_effect', 'decisions', 'dispatched_calls', 'truncated_responses']},
            failure_stages=dict(co.Counter(r['failure_stage'] for r in rows)))
    assert sum(x['heated'] for x in projected) == scored['action_counts_actual']['heated']
    assert sum(x['put_then_heat'] for x in projected) == scored['action_counts_actual']['put_then_heat']
    assert sum(x['physical_no_effect'] for x in projected) == scored['action_counts_actual']['physical_no_effect']
    observed_map = {r['trajectory_id']: r for r in observed['rows']}
    for r in records:
        assert r['success_curve'] == observed_map[r['trajectory_id']]['success_curve']
    for f, h in p['protected_files'].items():
        assert sha(f) == h
    audit_result = save(ROOT/'independent-analysis.json', dict(
        scope='Independent native-outcome, seed, window, shared-prefix, scene-interval and complete current-ledger reconstruction',
        protocol_sha256=p['sha256'], source_sha256=sha(__file__), frozen_result_sha256=scored['sha256'],
        protected_files_checked=95, all_checks_passed=True, actual_attempts=len(actual), base_trajectories=510,
        derived_trajectories=102, extra_fourth_attempts=extra, tasks=51, scenes=23,
        curves=curves, contrasts=contrasts, retention_fourth_pairs=dict(retention_pairs),
        action_by_condition=action_by_condition, actual_costs=actual_costs, ledger=dict(ledger),
        requests=len(requests), responses=len(responses), owned_services_absent=True,
        closeout_file_sha256=sha(ROOT/'allowance-closeout.json'), semantic_events=1349,
        unique_semantic_texts=len({r['text_sha256'] for r in q}), semantic_labels_completed=0,
        parameter_fits=0, new_model_calls=0, holdout134_calls=0, audit_elapsed_seconds=time.time()-start))
    # The public audit is byte-identical to the checked private result.
    (PUBLIC/'recovery-mechanism-audit-20261029.json').write_bytes((ROOT/'independent-analysis.json').read_bytes())
    save(PUBLIC/'recovery-mechanism-observed-20261029.json', dict(protocol_sha256=p['sha256'], rows=records,
        note='Derived correct_kept reuses first three corrected attempts; strategy costs include shared prefix, not additional physical calls'))
    for i, c in enumerate(CONDITIONS):
        rows = [r for r in projected if r['condition'] == c]
        save(PUBLIC/f'recovery-mechanism-attempts-{i}-20261029.json', dict(protocol_sha256=p['sha256'], condition=c, rows=rows))
    texts = {r['text_sha256']: r['text'] for r in q}
    reading = [dict(index=i, text_sha256=h, text=texts[h]) for i,h in enumerate(sorted(texts))]
    Path(ROOT/'semantic-reading-view.json').write_text(json.dumps(reading, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    save(ROOT/'semantic-progress.json', dict(queue_file_sha256=sha(ROOT/'failed-write-label-queue.json'),
        reading_file_sha256=sha(ROOT/'semantic-reading-view.json'), total_events=1349, total_unique_texts=1218,
        reviewed_unique_texts=0, decisions_file='semantic-decisions.json',
        status='pending_full_text_review', annotator='Single Codex, aggregate outcomes already known; not independent human or blind review'))
    print(json.dumps(dict(audit_sha256=audit_result['sha256'], curves=curves,
                          primary=[r for r in contrasts if r['primary']], retention_pairs=dict(retention_pairs),
                          elapsed=audit_result['audit_elapsed_seconds']), ensure_ascii=False))


if __name__ == '__main__':
    main()
