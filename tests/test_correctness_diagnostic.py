"""Check the five-way pairing, empty memory and committed native update boundary."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from agent_reliability.io import digest,write_json
from run_correctness_diagnostic import CONDITIONS,SEEDS,execute_cell,effect_summary,validate_matrix


def fixture():
    entries={'E':'Wrong premise.','C':'Correct premise.','N':'Generic plan.'}
    protocol={'entries':entries,'cases':[],'blocks':[],'seeds':list(SEEDS),
              'ports':[8001,8002,8003,8004],'gpu_indices':[0,1,2,3]}
    for i in range(12):
        task={'task_id':f'task-{i:02d}','scene_id':f'scene-{i:02d}','task_type':'pick_heat_then_place_in_recep'}
        protocol['cases'].append({'task_id':task['task_id'],'task':task,'initial_observation_sha256':digest('initial')})
        for seed in SEEDS:
            block={'task_id':task['task_id'],'seed':seed,'worker':i%4,'cells':[]}
            for condition,labels in CONDITIONS.items():
                block['cells'].append({'id':f'{i}-{seed}-{condition}','task_id':task['task_id'],
                    'seed':seed,'condition':condition,'labels':list(labels),
                    'memory_sha256':digest([entries[x] for x in labels])})
            protocol['blocks'].append(block)
    return protocol


class CorrectnessDiagnosticTests(unittest.TestCase):
    def test_matrix_preserves_empty_and_complete_paired_blocks(self):
        protocol=fixture();self.assertEqual(validate_matrix(protocol)['cells'],180)
        for kind in ['missing','duplicate','seed','empty_becomes_N','C_in_wrong_slot']:
            altered=copy.deepcopy(protocol)
            if kind=='missing':altered['blocks'].pop()
            elif kind=='duplicate':altered['blocks'][0]['cells'][1]=copy.deepcopy(altered['blocks'][0]['cells'][0])
            elif kind=='seed':altered['blocks'][0]['seed']=2103
            elif kind=='empty_becomes_N':altered['blocks'][0]['cells'][2]['labels']=['N']
            else:altered['blocks'][0]['cells'][3]['labels']=['C','N','E']
            with self.subTest(kind=kind),self.assertRaises(ValueError):validate_matrix(altered)

    def test_empty_attempt_and_native_failure_append_are_verified(self):
        protocol=fixture();case=protocol['cases'][0];cell=protocol['blocks'][0]['cells'][2];observed=[]
        def runner(task,memory,trial,directory,transport,data_root,seed,allowance,**kwargs):
            observed.append(list(memory));directory.mkdir(parents=True,exist_ok=True)
            identity={'task':task,'memory':memory,'trial':trial,'seed':seed,**kwargs};identity['sha256']=digest(identity)
            write_json(directory/'identity.json',identity);transitions=[{'event':'fixed'}]
            write_json(directory/'checkpoint.json',{'transitions':transitions})
            result={**task,'trial':trial,'memory_before':memory,'memory_after':memory+['New reflection.'],
                    'initial_observation_sha256':case['initial_observation_sha256'],'success':False,'decisions':49,
                    'identity_sha256':identity['sha256'],'trajectory_sha256':digest(transitions)}
            write_json(directory/'result.json',result);return result
        with tempfile.TemporaryDirectory() as temporary:
            row=execute_cell(Path(temporary),protocol,case,cell,None,lambda:None,runner)
        self.assertEqual(observed,[[]]);self.assertFalse(row['success'])
        self.assertEqual(row['memory_sha256'],digest([]))

    def test_effect_pairs_are_matched_by_task_seed_and_condition(self):
        protocol=fixture();rows=[]
        for block in protocol['blocks']:
            for cell in block['cells']:
                rows.append({'task_id':cell['task_id'],'seed':cell['seed'],'condition':cell['condition'],
                    'success':cell['condition'] in {'C','EC'}})
        result=effect_summary(rows)
        self.assertEqual(result['estimates']['EC_minus_EN']['mean'],1)
        self.assertEqual(result['estimates']['N_minus_empty']['mean'],0)
        rows[-1]=copy.deepcopy(rows[-2])
        with self.assertRaises(ValueError):effect_summary(rows)


if __name__=='__main__':unittest.main()
