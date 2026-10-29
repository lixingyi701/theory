#!/usr/bin/env python3
"""Persist full-text judgments and progress; never infer categories from keywords."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path('runs/reflexion-recovery-mechanism-20261029')
PUBLIC = Path('repro/reflexion-20261025')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save(p, x):
    x = dict(x, sha256=digest(x))
    Path(p).write_text(json.dumps(x, ensure_ascii=False, sort_keys=True, indent=2,
                                 allow_nan=False)+'\n', encoding='utf-8')
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['read', 'apply', 'status'])
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=24)
    args = ap.parse_args()
    view = read(ROOT/'semantic-reading-view.json')
    queue = read(ROOT/'failed-write-label-queue.json')
    audit = read(ROOT/'independent-analysis.json')
    assert len(view) == audit['unique_semantic_texts'] == 1218
    assert len(queue['rows']) == audit['semantic_events'] == 1349
    assert [r['index'] for r in view] == list(range(len(view)))
    for r in view:
        assert digest(r['text']) == r['text_sha256']
    if args.action == 'read':
        print(json.dumps(view[args.start:args.start+args.count], ensure_ascii=False))
        return
    if args.action == 'status':
        print((ROOT/'semantic-progress.json').read_text())
        return
    decisions = read(ROOT/'semantic-decisions.json')
    assert [r[0] for r in decisions] == list(range(len(decisions)))
    old_path = ROOT/'semantic-labels-partial.json'
    if old_path.exists():
        old = read(old_path)
        for entry in old['entries']:
            i = entry['index']
            assert [i, entry['category'], entry['evidence'], entry['reason']] == decisions[i], 'Preserve existing judgments; explicit correction required'
    entries = []
    for i, category, evidence, reason in decisions:
        row = view[i]
        assert category in ['E', 'C', 'B', 'U', 'A']
        assert evidence and all(q in row['text'] for q in evidence) and reason
        entries.append(dict(index=i, reflection_sha256=row['text_sha256'], category=category,
                            evidence=evidence, reason=reason))
    original = read(PUBLIC/'shared-calibration-semantic-labels.json')
    content = dict(scope='Full-text judgments in fixed hash order; partial until all1218 texts read',
        queue_file_sha256=sha(ROOT/'failed-write-label-queue.json'), reading_file_sha256=sha(ROOT/'semantic-reading-view.json'),
        decisions_file_sha256=sha(ROOT/'semantic-decisions.json'), category_definitions=original['category_definitions'],
        interpretation_rules=original['interpretation_rules'], entries=entries,
        total_unique_texts=1218, reviewed_unique_texts=len(entries),
        human_second_review=False, independent_blinded_annotation=False,
        annotator='Single Codex full-text reading and self-review; aggregate outcomes already known',
        new_model_calls=0, parameter_fits=0)
    labels = save(old_path, content)
    (PUBLIC/'recovery-mechanism-semantic-labels-20261029.json').write_bytes(old_path.read_bytes())
    reviewed = {r['reflection_sha256'] for r in entries}
    progress = save(ROOT/'semantic-progress.json', dict(
        queue_file_sha256=sha(ROOT/'failed-write-label-queue.json'), reading_file_sha256=sha(ROOT/'semantic-reading-view.json'),
        labels_sha256=labels['sha256'], total_events=1349, total_unique_texts=1218,
        reviewed_unique_texts=len(entries), covered_write_events=sum(r['text_sha256'] in reviewed for r in queue['rows']),
        next_index=len(entries), status='full_text_review_complete' if len(entries)==1218 else 'partial_full_text_review',
        next_step='Continue full-text labels before joint failed-and-E analysis',
        partial_category_counts_are_not_full_cohort_results=True,
        human_second_review=False, independent_blinded_annotation=False))
    (PUBLIC/'recovery-mechanism-semantic-progress-20261029.json').write_bytes((ROOT/'semantic-progress.json').read_bytes())
    print(json.dumps(progress, ensure_ascii=False))


if __name__ == '__main__':
    main()
