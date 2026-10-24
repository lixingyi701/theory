"""Completion adaptation must reject incomplete and duplicate evidence."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_correctness_diagnostic import compatible_summary
from test_correctness_diagnostic import fixture


class CorrectnessAuditTests(unittest.TestCase):
    def test_metadata_view_preserves_source_and_rejects_invalid_cohorts(self):
        protocol=fixture();protocol['sha256']='protocol'
        rows=[{'cell_id':c['id'],'task_id':c['task_id'],'seed':c['seed'],'condition':c['condition']}
              for b in protocol['blocks'] for c in b['cells']]
        summary={'protocol_sha256':'protocol','status':'complete','completed_cells':180,'expected_cells':180,
                 'worker_exit_codes':[0]*4,'worker_statuses':['complete']*4,'rows':rows}
        original=copy.deepcopy(summary)
        self.assertEqual(compatible_summary(summary,protocol)['completed_tasks'],12)
        self.assertEqual(summary,original)
        for kind in ['missing','duplicate','wrong_seed','worker_failed']:
            changed=copy.deepcopy(summary)
            if kind=='missing':changed['rows'].pop()
            elif kind=='duplicate':changed['rows'][0]=copy.deepcopy(changed['rows'][1])
            elif kind=='wrong_seed':changed['rows'][0]['seed']=2103
            else:changed['worker_exit_codes'][0]=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):compatible_summary(changed,protocol)


if __name__=='__main__':unittest.main()
