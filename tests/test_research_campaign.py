"""A continuation keeps prior costs and enforces its own wall-clock deadline."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from reflexion_campaign import permit


class CampaignTests(unittest.TestCase):
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
