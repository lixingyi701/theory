"""A continuation keeps prior costs and enforces its own wall-clock deadline."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from reflexion_campaign import permit
from run_reflexion_precondition_forks import memories


class CampaignTests(unittest.TestCase):
    def test_native_error_repair_preserves_non_target_text_and_controls(self):
        text = 'Plan: I should have placed it in the microwave, closed it, and used the heat command. Keep the final goal.'
        fragment = 'placed it in the microwave'; start = text.index(fragment)
        target = {'span': [start, start+len(fragment)], 'original_text': fragment,
                  'corrected_text': 'kept hold of it near the microwave'}
        parent = {'memory_after': ['older reflection', text]}
        branches = memories(parent, target)
        self.assertEqual(branches['original'], branches['sham'])
        self.assertEqual(branches['original'], branches['retain_error'])
        self.assertEqual(branches['discard_new'], ['older reflection'])
        self.assertEqual(branches['correct_target'][-1], text.replace(fragment, target['corrected_text'], 1))
        self.assertEqual(parent['memory_after'][-1], text)

    def test_cumulative_dispatch_and_deadline_limits(self):
        contract = {'prior_dispatched_calls': 9410, 'additional_max_calls': 6000, 'deadline_unix': 200}
        with patch('reflexion_campaign.ledger', return_value={'all': {'dispatched': 15409}}), \
                patch('reflexion_campaign.time.time', return_value=199):
            permit(contract)
        with patch('reflexion_campaign.ledger', return_value={'all': {'dispatched': 15410}}), \
                patch('reflexion_campaign.time.time', return_value=199):
            with self.assertRaisesRegex(RuntimeError, 'call limit'):
                permit(contract)
        with patch('reflexion_campaign.ledger', return_value={'all': {'dispatched': 9410}}), \
                patch('reflexion_campaign.time.time', return_value=200):
            with self.assertRaisesRegex(RuntimeError, 'deadline'):
                permit(contract)


if __name__ == '__main__':
    unittest.main()
