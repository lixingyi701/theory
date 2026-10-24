"""An extra digest is acceptable only when verified against the raw result."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from agent_reliability.io import digest
from close_native_clause import verify_worker_row


class NativeClauseCloseoutTests(unittest.TestCase):
    def test_extra_digest_is_verified_without_mutating_evidence(self):
        raw={'reflection':'stored text'};raw['sha256']=digest(raw)
        analysed={'panel':'critic','cell_id':'one','result_sha256':'file-digest'}
        worker={**analysed,'sha256':raw['sha256']}
        verify_worker_row(worker,analysed,raw)
        self.assertEqual(worker['sha256'],raw['sha256'])
        for changed_worker,changed_raw in [({**worker,'sha256':'wrong'},raw),
            ({**worker,'result_sha256':'other'},raw),(worker,{**raw,'reflection':'changed'})]:
            with self.assertRaises(ValueError):verify_worker_row(changed_worker,analysed,changed_raw)


if __name__=='__main__':unittest.main()
