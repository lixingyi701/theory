"""Critical checks for source-preserving edits and the paired two-panel matrix."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from agent_reliability.io import digest
from run_native_clause import critic_probe, edited_text, validate_matrix, SEEDS, CONDITIONS, PANELS


class NativeClauseTests(unittest.TestCase):
    def test_only_recorded_span_changes_and_source_mismatch_stops(self):
        original='Plan: A. Bad premise. B.'
        span='Bad premise.';start=original.index(span)
        record={'source':{'reflection_sha256':digest(original)},'edits':[
            {'start':start,'end':start+len(span),'source_span_sha256':digest(span),
             'correct_replacement':'Correct premise.','remove_replacement':''}],
             'variant_hashes':{'E':digest(original),'C':digest('Plan: A. Correct premise. B.'),'remove':digest('Plan: A.  B.')}}
        self.assertEqual(edited_text(original,record,'E'),original)
        self.assertEqual(edited_text(original,record,'C'),'Plan: A. Correct premise. B.')
        self.assertEqual(edited_text(original,record,'remove'),'Plan: A.  B.')
        with self.assertRaises(ValueError):edited_text(original+' ',record,'C')
        changed=deepcopy(record);changed['edits'][0]['end']+=1
        with self.assertRaises(ValueError):edited_text(original,changed,'C')

    def test_overlapping_edit_is_rejected(self):
        text='abcde';edit={'start':1,'end':3,'source_span_sha256':digest('bc'),'correct_replacement':'X','remove_replacement':''}
        record={'source':{'reflection_sha256':digest(text)},'edits':[edit,{**edit,'start':2}],
                'variant_hashes':{'E':digest(text)}}
        with self.assertRaises(ValueError):edited_text(text,record,'E')

    def test_matrix_requires_both_panels_and_all_pairings(self):
        tasks=[str(i) for i in range(12)]
        protocol={'cases':[{'task_id':t} for t in tasks],'seeds':SEEDS,'ports':[8001,8002,8003,8004],
                  'gpu_indices':[0,1,2,3],'blocks':[]}
        for i,task in enumerate(tasks):
            for seed in SEEDS:
                protocol['blocks'].append({'task_id':task,'seed':seed,'worker':i%4,'cells':[
                    {'id':f'{task}-{seed}-{panel}-{condition}','task_id':task,'seed':seed,'panel':panel,'condition':condition}
                    for panel in PANELS for condition in CONDITIONS]})
        resolver=lambda p,c:['one','two','three']
        self.assertEqual(validate_matrix(protocol,resolver)['actor_attempts'],108)
        for change in ['missing','duplicate','panel','seed']:
            altered=deepcopy(protocol);cells=altered['blocks'][0]['cells']
            if change=='missing':cells.pop()
            elif change=='duplicate':cells[0]=deepcopy(cells[1])
            elif change=='panel':cells[0]['panel']='other'
            else:cells[0]['seed']=2106
            with self.subTest(change=change),self.assertRaises(ValueError):validate_matrix(altered,resolver)

    def test_fixed_critic_uses_new_memory_and_original_failure_without_actor(self):
        class Provider:
            def __init__(self):self.calls=[]
            def complete(self,payload):
                self.calls.append(payload)
                return {'choices':[{'text':'Plan: keep trying','finish_reason':'stop'}],
                        'usage':{'prompt_tokens':1,'completion_tokens':1}}
        provider=Provider();reserved=[]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';source.mkdir()
            (source/'trial.log').write_text('OLD_MEMORY_MARKER\nHere is the task:\nPublic failure feedback.\n')
            memory=['N','NEW_MEMORY_MARKER','N']
            record={'fixed_failure':{'trial_log_sha256':'source-log'}}
            case={'task':{'task_id':'fixed-task'}}
            cell={'id':'probe','task_id':'fixed-task','seed':2107,'condition':'C'}
            with patch('run_native_clause.record_for',return_value=record),\
                 patch('run_native_clause.fixed_source',return_value=(source,{'decisions':7})),\
                 patch('run_native_clause.memory_for',return_value=memory):
                row=critic_probe(root,{},case,cell,provider,lambda:reserved.append(1))
            self.assertEqual(len(provider.calls),1);self.assertEqual(reserved,[1])
            self.assertIn('NEW_MEMORY_MARKER',provider.calls[0]['prompt'])
            self.assertIn('Public failure feedback.',provider.calls[0]['prompt'])
            self.assertNotIn('OLD_MEMORY_MARKER',provider.calls[0]['prompt'])
            self.assertEqual(row['panel'],'critic')
            self.assertTrue((root/'critic-cells/probe/trial-01/calls/reflector-0007/response.json').exists())


if __name__=='__main__':unittest.main()
