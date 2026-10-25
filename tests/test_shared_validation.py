"""Constructed scoring checks, separate from Agent measurements."""
import unittest
from evaluate_shared_validation import score_rows
from prepare_shared_calibration import STATES


class ValidationScoringTests(unittest.TestCase):
    def test_scene_paired_brier_and_incomplete_rejection(self):
        observed=[]
        for scene in range(9):
            for task in range(6):
                for condition in STATES:
                    observed.append({'scene_id':str(scene),'task_id':f'{scene}-{task}','condition':condition,
                                     'success_curve':[False,task<3]})
        forecasts={'selected':{c:[.1,.5] for c in STATES},'constant_persistent':{c:[.1,.1] for c in STATES},
                   'constant_pooled':{c:[.1,.1] for c in STATES},'error_pooled':{c:[.1,.1] for c in STATES}}
        report=score_rows({'forecasts':forecasts},observed)
        self.assertAlmostEqual(report['primary']['mean'],.16)
        self.assertEqual(report['primary_decision'],'prediction_improvement')
        with self.assertRaises(ValueError):score_rows({'forecasts':forecasts},observed[:-1])


if __name__=='__main__':unittest.main()
