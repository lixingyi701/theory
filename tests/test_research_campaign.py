"""A continuation keeps prior costs and enforces its own wall-clock deadline."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from reflexion_campaign import permit
from run_reflexion_precondition_forks import memories, repair_text
from analyze_reflexion_precondition_forks import procedure_events
from prepare_reflexion_precondition_targets import native_advice_spans


class CampaignTests(unittest.TestCase):
    def test_native_advice_is_distinct_from_history_or_negated_advice(self):
        wrong = 'After retrieving the cup, I should have placed it in the microwave, closed it, and used the heat command.'
        self.assertEqual(len(native_advice_spans(wrong, 'microwave')), 1)
        state_requirement = 'I should have first confirmed that the egg was in the microwave, then attempted to heat it.'
        self.assertEqual(len(native_advice_spans(state_requirement, 'microwave')), 1)
        abbreviated = 'I should have first confirmed the command format (e.g., "heat egg 1 in microwave 1") and ensured the egg was properly placed inside the microwave before heating.'
        proposals = native_advice_spans(abbreviated, 'microwave')
        self.assertTrue(any(p['text'].startswith('ensured') for p in proposals))
        for proposal in proposals:
            self.assertEqual(abbreviated[slice(*proposal['span'])], proposal['text'])
        for text in ['I placed the cup in the microwave and the command failed.',
                     'I should not place the cup in the microwave before heating.',
                     'I should have confirmed that the egg was not in the microwave before heating.',
                     'I should keep holding the cup rather than place it in the microwave.']:
            self.assertEqual(native_advice_spans(text, 'microwave'), [])

    def test_accepted_placement_and_retrieval_change_macro_state(self):
        steps = [
            {'action': 'take cup 1 from countertop 2', 'observation': 'You pick up the cup 1 from the countertop 2.'},
            {'action': 'put cup 1 in microwave 1', 'observation': 'You move the cup 1 to the microwave 1.'},
            {'action': 'heat cup 1 with microwave 1', 'observation': 'Nothing happens.'},
            {'action': 'take cup 1 from microwave 1', 'observation': 'You pick up the cup 1 from the microwave 1.'},
            {'action': 'close microwave 1', 'observation': 'You close the microwave 1.'},
            {'action': 'put cup 1 in microwave 1', 'observation': 'Nothing happens.'},
            {'action': 'heat cup 1 with microwave 1', 'observation': 'You heat the cup 1 using the microwave 1.'},
            {'action': 'heat apple 1 with microwave 1', 'observation': 'Nothing happens.'},
        ]
        events = procedure_events(steps, 'cup', 'heat', 'microwave 1')
        self.assertEqual([e['index'] for e in events['wrong_precondition_chains']], [2])
        self.assertEqual([e['index'] for e in events['held_successful_target_macros']], [6])
        self.assertEqual(events['held_successful_target_macros'][0]['preceding_rejected_placement_index'], 5)
        self.assertEqual([e['accepted'] for e in events['target_device_placement_attempts']], [True, False])
        self.assertEqual([e['index'] for e in events['competing_macros']], [7])

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
        repeated = 'I should have placed the mug in the microwave. Next I will place the mug in the microwave. Keep the stoveburner fallback.'
        changes = []
        for before, after in [('placed the mug in the microwave', 'kept the mug in my inventory near the microwave'),
                              ('place the mug in the microwave', 'keep the mug in my inventory near the microwave')]:
            index = repeated.index(before)
            changes.append({'span': [index, index+len(before)], 'original_text': before, 'corrected_text': after})
        repaired = repair_text(repeated, {'repairs': changes})
        self.assertNotIn('place the mug in the microwave', repaired)
        self.assertNotIn('placed the mug in the microwave', repaired)
        self.assertTrue(repaired.endswith('Keep the stoveburner fallback.'))
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            repair_text(repeated, {'repairs': [changes[0], changes[0]]})

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
