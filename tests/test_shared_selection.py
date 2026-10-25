"""Constructed optimizer/selection regression, not an Agent measurement."""
import unittest
import numpy as np
from prepare_shared_calibration import STATES
from shared_recovery_model import compare,component_kernel,features


class SharedSelectionTests(unittest.TestCase):
    def test_separate_write_selection_and_assembled_kernel(self):
        rows=[]
        for i in range(12):
            for condition,labels in STATES.items():
                state=[{'type':k,'native':False} for k in labels]
                success='E' not in labels or labels[-1]=='C'
                rows.append({'scene_id':str(i//2),'task_id':str(i),
                    'trajectory_id':f'{i}-{condition}','trial':1,'state':state,'outcome':'S' if success else 'E'})
                if not success:
                    rows.append({**rows[-1],'trial':2,'state':(state+[{'type':'E','native':True}])[-3:]})
        result=compare(rows)
        self.assertEqual(result['selected_write']['family'],'error_lifetime')
        self.assertEqual(result['selected_success']['family'],'information_order')
        self.assertEqual(len(result['folds']),36)
        selected=result['selected_model']
        self.assertEqual(len(selected['write_coefficients_reference_A']),len(features(rows[0]['state'],'error_lifetime')))
        self.assertEqual(len(selected['success_coefficients']),len(features(rows[0]['state'],'information_order')))
        kernel=component_kernel(selected,rows[0]['state'])
        self.assertTrue(np.isfinite(kernel).all())
        np.testing.assert_allclose(kernel.sum(axis=1),1.,atol=1e-12)


if __name__=='__main__': unittest.main()
